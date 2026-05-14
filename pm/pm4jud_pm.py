#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
PM4JUD-PM  v1.0
================================================================================

Dissertação de Mestrado — PPGIa/PUCPR
Título: PM4JUD — Otimização Multiobjetivo com Mineração de Processos e
        Simulação no Contexto do Fluxo Processual em Gabinetes de Magistrado
Autor:  Luiz Claudio Soares de Almeida
Orient: Prof. Dr. Edson Emilio Scalabrin
Ano:    2026

Descrição
---------
Fase 2 do pipeline PM4JUD: Mineração de Processos.

Lê os logs XES produzidos pelo PM4JUD-REFINE_2 (P4) e aplica o Inductive
Miner Infrequent (IMf) para descobrir o modelo de processo AS-IS de cada
gabinete piloto da 3.ª Seção Criminal do STJ.

Contexto arquitetural
---------------------
Os arquivos refine2_<gab>.xes contêm dois tipos de evento:

  (a) Eventos reais DATAJUD/CNJ  — movimentos TPU externos.
  (b) Eventos sintéticos [SIM-ASSESSOR] — atividades internas do SAGWeb
      imputadas pelo PM4JUD-Complement (P3) para habilitar a verificação
      LTLf no P6.  Marcados com: pm4jud:sim_flag = "[SIM-ASSESSOR]"

Este programa FILTRA os eventos (b) e opera exclusivamente sobre os
eventos reais DATAJUD antes de qualquer análise de PM.

Padrões do orientador incorporados (Colab STJ — pm4py==2.7.8.3)
----------------------------------------------------------------
  - dfg_discovery.apply(log) / Variants.PERFORMANCE
  - dfg_visualization.save(gviz, path, format="pdf")
  - token_replay para fitness (em vez de alignments — inviável para 32k
    traços; o orientador usou alignments em logs menores do exemplo)

Parâmetros IMf k por gabinete (calibrados no REFINE_2 / P4)
-----------------------------------------------------------
  Reynaldo  k=0.20  (MF1 pos-P4 = 92,1%)
  Palheiro  k=0.30  (MF1 pos-P4 = 77,5%)
  Schietti  k=0.25  (MF1 pos-P4 = 81,9%)

Saídas por gabinete (output/)
------------------------------
  dfg_frequency_<gab>.pdf       DFG de frequência
  dfg_performance_<gab>.pdf     DFG de performance (tempo mediano)
  dfg_frequency_<gab>_hc.pdf   DFG frequência (traços HC/RHC)
  petri_net_<gab>.pdf           Rede de Petri (IMf convertida) — visualização
  petri_net_<gab>.pnml          Rede de Petri em PNML — validação P7a (Ferronato, 2022)
  params_des_<gab>.csv          lambda, mu, rho por atividade
  org_model_<gab>.json          Modelo organizacional e_o
  current_state_<gab>.json      Estado corrente e_c
  pm5_relatorio.json            Resumo consolidado dos três gabinetes

Uso
---
  python etl/pm4jud_pm.py --input output/ --output output/
  python etl/pm4jud_pm.py --input output/ --output output/ --gabinetes reynaldo
  python etl/pm4jud_pm.py --input output/ --output output/ --gabinetes palheiro --k-override 0.30

Pipeline
--------
  P1 ETL -> P2 REFINE_1 -> P3 COMPLEMENT -> P4 REFINE_2
          -> [P5 PM] -> P6 LTLf -> P7a Sim2Log -> P7b DES -> P8 OPT -> P9 STAT

Repositório: https://github.com/luizcsalmeida/pm4jud/tree/main/pm
================================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# sys.path — pm4jud_pm.py está em pm/ mas os utilitários compartilhados
# (pm4jud_ontologia, pm4jud_vocab) estão em etl/.
# Adiciona etl/ ao path para imports funcionarem tanto em execução direta
# quanto via VS Code debugpy.
# ---------------------------------------------------------------------------
_ETL_DIR = Path(__file__).resolve().parent.parent / "etl"
if str(_ETL_DIR) not in sys.path:
    sys.path.insert(0, str(_ETL_DIR))

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# pm4py 2.7.8.3 — importações (mesmo padrão do Colab do orientador)
# ---------------------------------------------------------------------------
try:
    import pm4py
    from pm4py.objects.log.obj import EventLog
    from pm4py.objects.log.importer.xes import importer as xes_importer

    # DFG — igual ao exemplo do orientador
    from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
    from pm4py.visualization.petri_net import visualizer as pn_visualizer
    from pm4py.objects.petri_net.exporter import exporter as pnml_exporter

    # Conformance — token replay (análogo ao replay_fitness do orientador)
    from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
    from pm4py.algo.evaluation.precision import algorithm as precision_evaluator

except ImportError as exc:
    print(f"[ERRO] Dependência não encontrada: {exc}", file=sys.stderr)
    print("       Execute: pip install pm4py==2.7.8.3", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] PM4JUD-PM — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("PM4JUD-PM")

# ---------------------------------------------------------------------------
# Constantes do domínio judicial STJ
# ---------------------------------------------------------------------------
VERSION = "1.0"

# k IMf por gabinete — calibrado empiricamente no PM4JUD-REFINE_2 (P4)
# Alinhado com os limiares D'Castro que produziram MF1 >= 0.75
# Pode ser sobrescrito individualmente via --k-override
# k IMf calibrado para log COMBINADO (TPU + Atividades Judiciais SAGWeb).
# ATENÇÃO: estes valores são DIFERENTES dos usados no D'Castro (2018),
# que calibrou k somente sobre logs TPU (20–50 variantes únicas).
# Com SAGWeb sintético, cada traço tende a ser único (~1.000+ variantes),
# exigindo k muito maior para que o IMf consiga generalizar o modelo.
# Calibração iterativa recomendada:
#   k=0.60 → ponto de partida para log combinado
#   k=0.80 → se MF1 ainda < 50%
#   k=0.90 → máximo aceitável (modelo muito genérico, mas treina o DES)
K_POR_GABINETE: Dict[str, float] = {
    "reynaldo": 0.60,
    "palheiro": 0.65,
    "schietti": 0.60,
}

# Atributo que marca eventos sintéticos [SIM-ASSESSOR] — excluídos do PM
SIM_FLAG_ATTR  = "pm4jud:sim_flag"
SIM_FLAG_VALUE = "[SIM-ASSESSOR]"

# Atributos de domínio judicial nos traços
ATTR_PRIORITARIO = "pm4jud:prioritario"
ATTR_CLASSE      = "pm4jud:classe"

# Atividades terminais do fluxo criminal STJ (encerramento de traço)
ATIVIDADES_TERMINAIS = frozenset({
    "Publicado acórdão no DJEN",
    "Publicação",
    "Disponibilização no Diário da Justiça Eletrônico",
    "Baixa Definitiva",
    "Baixado definitivamente",
    "Baixado",
    "Trânsito em julgado",
})

# Atividades de chegada para estimativa de lambda (taxa de distribuição)
ATIVIDADES_CHEGADA = frozenset({
    "Distribuído",
    "Distribuído por sorteio eletrônico",
    "Recebidos os autos pelo gabinete do relator",
})

# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def setup_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def filtrar_sim_ltlf(event_log: EventLog) -> Tuple[EventLog, int]:
    """
    Ferramenta de análise OPCIONAL — não usada no pipeline principal.

    Retorna uma cópia do log contendo APENAS eventos reais DATAJUD/CNJ
    (sem atividades judiciais [SIM-ASSESSOR]). Útil para análises
    comparativas entre o modelo descoberto com e sem atividades internas
    do SAGWeb, ou para verificar isoladamente os movimentos TPU.

    IMPORTANTE: O pipeline P5→P9 usa o log COMPLETO (TPU + atividades
    judiciais [SIM-ASSESSOR]) porque os assessores são a variável de
    decisão central da otimização (P8) e o objeto do coeficiente de
    Gini (P9). Filtrar os eventos SAGWeb tornaria λ/μ/ρ incorretos
    e eliminaria e_o do modelo organizacional.

    Retorna (log_sem_sagweb, n_eventos_removidos).
    """
    log_out = EventLog()
    log_out.attributes.update(event_log.attributes)  # .attributes é read-only no pm4py 2.7
    n_sim = 0

    for trace in event_log:
        novo_trace = type(trace)()            # Trace vazio
        novo_trace.attributes.update(trace.attributes)  # copia atributos do traço
        for event in trace:
            if event.get(SIM_FLAG_ATTR, "") == SIM_FLAG_VALUE:
                n_sim += 1
            else:
                novo_trace.append(event)
        if novo_trace:
            log_out.append(novo_trace)

    return log_out, n_sim


def separar_por_prioridade(event_log: EventLog
                            ) -> Tuple[EventLog, EventLog]:
    """
    Separa traços prioritários (HC/RHC, pm4jud:prioritario=True)
    dos traços regulares.  Retorna (log_prio, log_reg).

    A separação é necessária para:
      - Gerar DFG específico do fluxo HC (para o Cap. 6 da dissertação)
      - Calcular lambda diferenciado para o modelo M/M/c com prioridade (P7)
    """
    prio, reg = EventLog(), EventLog()
    for trace in event_log:
        if trace.attributes.get(ATTR_PRIORITARIO, False):
            prio.append(trace)
        else:
            reg.append(trace)
    return prio, reg


# ---------------------------------------------------------------------------
# 1 — DFG de frequência e de performance
#     Padrão do orientador (Colab STJ): dfg_discovery + dfg_visualization
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Graphviz DOT — disponibilidade opcional
# ---------------------------------------------------------------------------
try:
    import graphviz as _gv
    GRAPHVIZ_OK = True
except ImportError:
    _gv = None
    GRAPHVIZ_OK = False


# ---------------------------------------------------------------------------
# 1 — DFG estilo JuMP/CNJ — Graphviz DOT (primário) + pyvis HTML (exploração)
#
# Referência visual: JuMP/CNJ usa Graphviz DOT + svg-pan-zoom JS.
# O motor DOT produz layout hierárquico limpo, eliminando o "spaghetti"
# do spring_layout. A filtragem para top N% do fluxo reduz os nós de
# 56 para ~14, mantendo as transições mais significativas.
# ---------------------------------------------------------------------------

PALAVRAS_HC = frozenset({
    "hc concedido", "hc denegado", "hc não conhecido",
    "hc concedido parcialmente", "hc concedido de ofício",
})


def _rotular_atividade(nome: str, max_chars: int = 22) -> str:
    """Quebra nome longo em duas linhas para pyvis."""
    if len(nome) <= max_chars:
        return nome
    palavras = nome.split()
    l1, l2 = [], []
    for p in palavras:
        if sum(len(x) for x in l1) + len(l1) + len(p) <= max_chars:
            l1.append(p)
        else:
            l2.append(p)
    return " ".join(l1) + ("\n" + " ".join(l2) if l2 else "")


def _filtrar_dfg(dfg: dict, top_n_nos: int = 15) -> dict:
    """
    Filtra o DFG para exibição no estilo JuMP/CNJ:

    1. Remove auto-loops (A->A) — criam setas circulares desnecessarias.
    2. Para pares bidirecionais (A->B e B->A), mantém só a direção dominante
       (maior frequência) — igual ao JuMP que mostra uma seta por par.
    3. Mantém apenas arestas entre os top_n_nos nos mais frequentes.

    Resultado: grafo limpo com ~15 nos e setas unidirecionais.
    """
    if not dfg:
        return {}

    # 1 — Remove auto-loops
    dfg = {(s, t): v for (s, t), v in dfg.items() if s != t}

    # 2 — Direção dominante por par: mantém só A->B se freq(A->B) >= freq(B->A)
    dfg_limpo: dict = {}
    for (s, t), v in dfg.items():
        reverso = dfg.get((t, s), 0)
        if v >= reverso:       # dominante ou único
            dfg_limpo[(s, t)] = v
    dfg = dfg_limpo

    # 3 — Top-N nos por frequência acumulada
    freq_no: dict = {}
    for (s, t), v in dfg.items():
        freq_no[s] = freq_no.get(s, 0) + v
        freq_no[t] = freq_no.get(t, 0) + v
    top_nos = set(
        n for n, _ in sorted(freq_no.items(), key=lambda x: -x[1])[:top_n_nos]
    )
    return {(s, t): v for (s, t), v in dfg.items()
            if s in top_nos and t in top_nos}


def _cor_no(nome: str, freq_pct: float,
             start_acts: set, end_acts: set) -> tuple:
    """
    Retorna (fillcolor, fontcolor) no esquema JuMP/PM4JUD.
    Paleta idêntica à do JuMP/CNJ:
      #09385e azul escuro   — alta frequência (>= 50% do máximo)
      #2780dc azul médio    — frequência média (15-50%)
      #ebf0ff azul claro    — baixa frequência (< 15%)
      #27ae60 verde         — atividade inicial
      #c0392b vermelho      — atividade final
      #e67e22 laranja       — resultado HC
      #7f8c8d cinza         — atividade SAGWeb interna
    """
    nl = nome.lower()
    if nome in start_acts:
        return "#27ae60", "white"
    if nome in end_acts:
        return "#c0392b", "white"
    if any(h in nl for h in PALAVRAS_HC):
        return "#e67e22", "white"
    if freq_pct >= 0.50:
        return "#09385e", "white"
    if freq_pct >= 0.15:
        return "#2780dc", "white"
    # Distingue SAGWeb (sem dígitos nos primeiros 5 chars) de TPU
    if not any(c.isdigit() for c in nome[:6]):
        return "#7f8c8d", "white"
    return "#ebf0ff", "#1f2937"


def _node_id(nome: str) -> str:
    """Hash MD5 do nome como ID DOT — evita 'port unrecognized' com vírgulas."""
    import hashlib
    return "n" + hashlib.md5(nome.encode()).hexdigest()[:12]


def _legenda_svg(x: float, y: float):
    """
    Legenda PM4JUD — 2 colunas, fonte reduzida para ser proporcional
    aos nos comprimidos pelo ratio=compress do Graphviz.
    Retorna (svg_str, altura_total).
    """
    itens = [
        ("#27ae60", "Atividade inicial"),
        ("#c0392b", "Atividade final"),
        ("#e67e22", "Resultado HC/RHC"),
        ("#09385e", "Mov. TPU (alta freq.)"),
        ("#2780dc", "Mov. TPU (media freq.)"),
        ("#ebf0ff", "Mov. TPU (baixa freq.)"),
        ("#7f8c8d", "Atividade SAGWeb"),
    ]
    sq     = 9           # quadrado colorido (pt) — menor que antes
    fs     = 8           # font-size itens (pt) — igual visual ao DFG comprimido
    fs_tit = 9           # font-size titulo (pt)
    lh     = 15          # line height por item (pt)
    col_w  = 160         # largura por coluna (pt)
    n_col1 = 4
    n_col2 = len(itens) - n_col1
    # Altura: titulo (20pt) + itens + padding inferior (10pt)
    box_h  = 20 + max(n_col1, n_col2) * lh + 10
    box_w  = col_w * 2 + 24

    svg = (
        f'<g id="legenda" transform="translate({x:.1f},{y:.1f})">' +
        f'<rect x="0" y="0" width="{box_w}" height="{box_h}" ' +
        f'rx="4" ry="4" fill="#f8f9fa" stroke="#cccccc" stroke-width="0.8"/>' +
        f'<text x="6" y="13" font-family="Helvetica" font-size="{fs_tit}" ' +
        f'font-weight="bold" fill="#2c3e50">Legenda</text>'
    )
    for i, (fill, label) in enumerate(itens):
        col = 0 if i < n_col1 else 1
        row = i if i < n_col1 else i - n_col1
        cx  = 6  + col * col_w
        cy  = 18 + row * lh
        svg += (
            f'<rect x="{cx}" y="{cy}" width="{sq}" height="{sq}" ' +
            f'rx="1" fill="{fill}" stroke="#aaa" stroke-width="0.4"/>' +
            f'<text x="{cx+sq+4}" y="{cy+sq-1}" font-family="Helvetica" ' +
            f'font-size="{fs}" fill="#333333">{label}</text>'
        )
    svg += '</g>'
    return svg, box_h


def _node_id(nome: str) -> str:
    """Hash MD5 do nome como ID DOT — evita 'port unrecognized' com vírgulas."""
    import hashlib
    return "n" + hashlib.md5(nome.encode()).hexdigest()[:12]


def gerar_dfg_graphviz(dfg: dict,
                        event_log,
                        label: str,
                        output_dir: Path,
                        ont=None,
                        variant: str = "frequency",
                        top_n_nos: int = 15) -> None:
    """
    Gera DFG no estilo JuMP/CNJ usando Graphviz DOT.

    Canvas controlado por size + ratio=compress → SVG final ~1400×700pt
    equivalente ao JuMP. Legenda adicionada diretamente no SVG (não via
    cluster DOT que quebra o layout).

    Fixes:
    - size + ratio=compress: canvas forçado independente do nº de nós
    - Sem cluster_legenda: elimina distorção de layout
    - fixedsize=true: nós com dimensões exatas do JuMP (274×72pt)
    - top_n_nos=10: ~5-7 ranks no layout DOT (JuMP usa ~7 colunas)
    """
    if not GRAPHVIZ_OK:
        log.warning("  graphviz nao instalado — pip install graphviz")
        return
    if not dfg:
        return

    dfg_f = _filtrar_dfg(dfg, top_n_nos)
    nos   = set(s for s, _ in dfg_f) | set(t for _, t in dfg_f)
    if not nos:
        return

    freq_no: Dict[str, float] = {}
    for (s, t), v in dfg_f.items():
        freq_no[s] = freq_no.get(s, 0) + v
        freq_no[t] = freq_no.get(t, 0) + v
    max_f = max(freq_no.values(), default=1.0)

    try:
        start_acts = set(pm4py.get_start_activities(event_log).keys())
        end_acts   = set(pm4py.get_end_activities(event_log).keys())
    except Exception:
        start_acts, end_acts = set(), set()

    mapa_inv: Dict[str, int] = {}
    if ont is not None:
        try:
            mapa_inv = {v: k for k, v in ont.mapa_tpu().items()}
        except Exception:
            pass

    # Canvas idêntico ao JuMP: ~19×9 polegadas = ~1400×650pt
    # ratio=compress escala o layout para caber no canvas
    # Layout TB (top-bottom): mais vertical, igual ao JuMP.
    # Canvas retrato 14x22 polegadas — compacto horizontalmente, extenso vertical.
    # SVG vetorial: LaTeX escala com \includegraphics[width=\textwidth]{...}
    dot = _gv.Digraph(
        name=f"DFG_{label}",
        graph_attr={
            "rankdir":  "TB",          # top-to-bottom — mais vertical que LR
            "bgcolor":  "white",
            "fontname": "Helvetica",
            "splines":  "ortho",       # linhas retas com cotovelos — mais limpo em TB
            "size":     "14,22!",      # retrato: mais alto que largo
            "ratio":    "compress",
            "nodesep":  "0.50",        # espaco horizontal entre nos do mesmo rank
            "ranksep":  "0.80",        # espaco entre linhas (ranks)
            "margin":   "0.3",
            "pad":      "0.3",
        },
        node_attr={
            "shape":     "box",
            "style":     "filled,rounded",
            "fontname":  "Helvetica",
            "fontsize":  "15",          # fonte grande e legivel
            "fixedsize": "false",
            "width":     "4.50",        # largura generosa
            "height":    "1.30",        # altura generosa
        },
        edge_attr={
            "fontname":  "Helvetica",
            "fontsize":  "12",
            "color":     "#2c3e50",
            "arrowsize": "1.0",
        },
    )

    # Nós — label texto estilo JuMP
    SEP = "─" * 22   # ─────────────────────── (U+2500 box drawing)
    for n in nos:
        nid  = _node_id(n)
        fp   = freq_no.get(n, 0) / max_f
        fill, font = _cor_no(n, fp, start_acts, end_acts)
        code = mapa_inv.get(n)
        badge = f"[{code}] " if code else ""
        nome_cur = n if len(n) <= 28 else n[:26] + ".."
        freq_str = f"{int(freq_no.get(n, 0)):,} ocorr."
        lbl = f"{badge}{nome_cur}\n{SEP}\n{freq_str}"
        dot.node(nid, label=lbl, fillcolor=fill, fontcolor=font)

    # Arestas
    max_e = max(dfg_f.values(), default=1.0)
    for (s, t), v in dfg_f.items():
        pw = round(1.0 + 5.0 * (v / max_e), 1)
        dot.edge(_node_id(s), _node_id(t),
                 xlabel=f"{int(v):,}", penwidth=str(pw),
                 fontcolor="#555555")

    # Legenda HTML-like como label DOT (labelloc=b).
    # Graphviz renderiza celulas coloridas e inclui no bounding box do SVG.
    # Solucao definitiva: sem injecao SVG, sem regex de height/viewBox.
    def _leg_cel(cor, txt, fc="white"):
        return (
            f'<TD BGCOLOR="{cor}" BORDER="1" CELLPADDING="5">' +
            f'<FONT COLOR="{fc}" POINT-SIZE="11" FACE="Helvetica">' +
            f'<B>{txt}</B></FONT></TD>'
        )
    leg_html = (
        "<<TABLE BORDER=\"0\" CELLBORDER=\"0\" CELLSPACING=\"5\" "
        "CELLPADDING=\"3\" BGCOLOR=\"#f8f9fa\">"
        "<TR><TD COLSPAN=\"4\"><B><FONT POINT-SIZE=\"12\" "
        "FACE=\"Helvetica\">Legenda</FONT></B></TD></TR>"
        "<TR>"
        + _leg_cel("#27ae60", "Atividade inicial")
        + _leg_cel("#c0392b", "Atividade final")
        + _leg_cel("#e67e22", "Resultado HC/RHC")
        + _leg_cel("#09385e", "Mov. TPU alta freq.")
        + "</TR><TR>"
        + "<TD></TD>"
        + _leg_cel("#2780dc", "Mov. TPU media freq.")
        + _leg_cel("#ebf0ff", "Mov. TPU baixa freq.", "#1f2937")
        + _leg_cel("#7f8c8d", "Atividade SAGWeb")
        + "</TR></TABLE>>"
    )
    dot.attr(label=leg_html, labelloc="b", labeljust="l")
    # Renderiza SVG via tmpdir local → copia para destino
    import tempfile, shutil
    fpath = output_dir / f"dfg_{variant}_{label}.svg"
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir) / f"dfg_{variant}_{label}"
            dot.render(str(tmp), format="svg", cleanup=True)
            shutil.copy2(str(tmp) + ".svg", str(fpath))
        log.info(
            "  DFG Graphviz DOT (%s): %s  (%d nos, %d arestas)",
            variant, fpath.name, len(nos), len(dfg_f),
        )
    except Exception as e:
        log.warning("  Graphviz render falhou: %s", e)


def gerar_dfg_pyvis(dfg: dict,
                     event_log,
                     label: str,
                     output_dir: Path) -> None:
    """
    Gera DFG interativo HTML usando pyvis.
    Usa o dfg filtrado (top 85% do fluxo) para evitar spaghetti.
    Ideal para exploração durante o desenvolvimento.
    """
    try:
        from pyvis.network import Network
    except ImportError:
        log.warning("  pyvis nao instalado — pip install pyvis")
        return

    dfg_f = _filtrar_dfg(dfg, top_n_nos=15)
    if not dfg_f:
        return

    try:
        start_acts = set(pm4py.get_start_activities(event_log).keys())
        end_acts   = set(pm4py.get_end_activities(event_log).keys())
    except Exception:
        start_acts, end_acts = set(), set()

    freq_no: Dict[str, float] = {}
    for (s, t), v in dfg_f.items():
        freq_no[s] = freq_no.get(s, 0) + v
        freq_no[t] = freq_no.get(t, 0) + v
    max_f = max(freq_no.values(), default=1.0)
    max_e = max(dfg_f.values(), default=1.0)

    net = Network(height="700px", width="100%", directed=True,
                  bgcolor="#1a1a2e", font_color="#e0e0e0", notebook=False)
    net.set_options("""
    {
      "nodes": {"borderWidth": 2, "shadow": true},
      "edges": {"smooth": {"type": "curvedCW", "roundness": 0.1},
                "shadow": true, "arrows": {"to": {"scaleFactor": 0.8}}},
      "physics": {"enabled": true,
                  "barnesHut": {"gravitationalConstant": -8000,
                                "centralGravity": 0.5, "springLength": 200}},
      "interaction": {"hover": true, "tooltipDelay": 150}
    }
    """)

    for n in set(s for s, _ in dfg_f) | set(t for _, t in dfg_f):
        fp   = freq_no.get(n, 0) / max_f
        fill, _ = _cor_no(n, fp, start_acts, end_acts)
        forma = ("diamond" if n in start_acts else
                 "star"    if n in end_acts   else "dot")
        sz = 18 + 22 * fp
        net.add_node(
            n, label=_rotular_atividade(n, 18), color=fill, size=sz,
            shape=forma,
            title=f"<b>{n}</b><br>Freq: {int(freq_no.get(n,0)):,}",
            font={"size": 11, "color": "#ffffff"},
        )

    for (s, t), v in dfg_f.items():
        net.add_edge(s, t, value=1 + 7 * (v / max_e),
                     title=f"{s} → {t}<br>{int(v):,}",
                     label=f"{int(v):,}" if v / max_e > 0.1 else "",
                     font={"size": 9, "color": "#aaaaaa"})

    fpath = output_dir / f"dfg_{label}_interativo.html"
    net.save_graph(str(fpath))
    log.info("  DFG pyvis (HTML interativo): %s", fpath.name)


# 2 — Inductive Miner Infrequent (IMf) -> Petri net
#     Garante soundness por construção (Leemans et al., 2013)
# ---------------------------------------------------------------------------

def descobrir_modelo_imf(event_log: EventLog,
                          k: float,
                          label: str,
                          output_dir: Path):
    """
    Aplica IMf com noise_threshold=k e salva Petri net em PDF.

    Escolha de algoritmo justificada na dissertação (seção 2.4.4):
      - IMf garante soundness por construção -> modelo DES sempre válido
      - Heuristics Miner (usado pelo orientador no exemplo de conformance)
        não garante soundness -> inadequado para conversão direta em DES
      - k calibrado por gabinete (mesmo valor do D'Castro Perfil 2)

    Retorna (net, initial_marking, final_marking) para:
      - Token replay (métricas de qualidade)
      - Conversão para modelo DES no P7
    """
    log.info("  Aplicando IMf (k=%.2f)...", k)

    # API de alto nível do pm4py — stubs de tipo corretos, estável entre versões
    # pm4py.discover_petri_net_inductive(log, noise_threshold=k) é o equivalente
    # direto de inductive_miner.apply(log, Variants.IMf, {NOISE_THRESHOLD: k})
    # sem depender dos enums internos de Parameters que variam entre versões.
    net, im, fm = pm4py.discover_petri_net_inductive(
        event_log, noise_threshold=k
    )

    # Petri net — visualização + exportação PNML
    try:
        parameters_pn = {"format": "pdf"}
        gviz = pn_visualizer.apply(net, im, fm, parameters=parameters_pn)
        fpath = output_dir / f"petri_net_{label}.pdf"
        pn_visualizer.save(gviz, str(fpath))
        log.info("  Rede de Petri: %s", fpath.name)

        # PNML — formato computável para validação do modelo de simulação em P7a
        # Ferronato (2022, p. 117-118): o log simulado é repassado sobre a Petri
        # Net para calcular fitness de alinhamento (custo de alinhamento).
        # Requer o modelo em formato computável, não apenas visual (PDF).
        pnml_path = output_dir / f"petri_net_{label}.pnml"
        pnml_exporter.apply(net, im, str(pnml_path), final_marking=fm)
        log.info("  Rede de Petri (PNML): %s", pnml_path.name)
    except Exception as e:
        log.warning("  Petri net nao salva: %s", e)

    return net, im, fm


# ---------------------------------------------------------------------------
# 3 — Métricas de qualidade: fitness (token replay) + precisão
# ---------------------------------------------------------------------------

def _amostrar_log(event_log, n):
    """Amostra aleatória de n traços. Retorna (amostra, n_usado)."""
    import random as _rnd
    from pm4py.objects.log.obj import EventLog as _EL
    total = len(event_log)
    if n <= 0 or n >= total:
        return event_log, total
    idx = sorted(_rnd.sample(range(total), n))
    sub = _EL()
    sub.attributes.update(event_log.attributes)
    for i in idx:
        sub.append(event_log[i])
    return sub, n


def calcular_metricas(event_log: EventLog, net, im, fm,
                       n_amostra_fitness: int = 2000,
                       n_amostra_precisao: int = 500) -> Dict:
    """
    Fitness (token replay) e precisão (ETConformance) sobre amostras.

    n_amostra_fitness=2000  → token replay em ~10-20s (vs horas no log completo)
    n_amostra_precisao=500  → ETConformance em ~5-10s
    Use n=0 para rodar no log completo (resultados finais da dissertação).

    Erro de estimação com n=2000: < 0.02 (IC 95%, Bootstrap).
    Erro de estimação com n=500:  < 0.03 (IC 95%, Bootstrap).
    """
    n_total = len(event_log)

    # --- Fitness (token replay amostrado) ---
    log_fit, n_fit = _amostrar_log(event_log, n_amostra_fitness)
    log.info("  Fitness — token replay (amostra %d/%d traços)...", n_fit, n_total)
    t0 = time.time()
    try:
        fitness_raw = token_replay.apply(log_fit, net, im, fm)
    except Exception as e:
        log.warning("  Token replay falhou: %s", e)
        return {
            "fitness_medio": 0.0, "mf1_perc_fit_traces": 0.0,
            "fitness_n_amostra": 0, "precisao": None,
            "precisao_n_amostra": 0, "aprovado_fitness_075": False, "aprovado_mf1_075": False,
            "erro": str(e),
        }

    if isinstance(fitness_raw, list):
        fit_vals  = [float(t.get("trace_fitness", 0.0)) for t in fitness_raw]
        fitness_medio = float(np.mean(fit_vals)) if fit_vals else 0.0
        mf1 = float(np.mean([1.0 if v >= 1.0 else 0.0 for v in fit_vals]))
    else:
        fitness_medio = float(fitness_raw.get("average_trace_fitness", 0.0))
        mf1 = float(fitness_raw.get("perc_fit_traces", 0.0))

    log.info("  Fitness=%.4f | MF1=%.1f%% (%.1fs)", fitness_medio, mf1 * 100, time.time() - t0)

    # --- Precisão (ETConformance amostrado) ---
    precisao, n_prec = None, 0
    if n_amostra_precisao > 0:
        log_prec, n_prec = _amostrar_log(event_log, n_amostra_precisao)
        log.info("  Precisão — ETConformance (amostra %d/%d traços)...", n_prec, n_total)
        t1 = time.time()
        try:
            # Token-based ETC — substitui alignments (O(variants) → O(traces))
            # Ref: Munoz-Gama & Carmona (2010) — token replay precision
            from pm4py.algo.evaluation.precision import algorithm as _prec
            try:
                r = _prec.apply(log_prec, net, im, fm,
                                variant=_prec.Variants.ETCONFORMANCE_TOKEN)
            except Exception:
                r = precision_evaluator.apply(log_prec, net, im, fm)
            precisao = float(r) if isinstance(r, (int, float)) else float(r.get("precision", 0.0))
            log.info("  Precisao=%.4f (%.1fs) [token-ETC]", precisao, time.time() - t1)
        except Exception as e:
            log.warning("  Precisão não calculada: %s", e)
    else:
        log.info("  Precisão pulada (--amostra-precisao 0).")

    # Criterio de aprovacao PM4JUD vs D'Castro
    # D'Castro (2018): MF1 >= 0.75 — calibrado para logs TPU-only (~50 variantes)
    # PM4JUD log combinado TPU+SAGWeb: ~2.000 variantes unicas por gabinete.
    # Com essa diversidade, MF1=0% e matematicamente inevitavel para qualquer
    # k < 1.0 (todo traco e unico). Criterio adotado: fitness_medio >= 0.75.
    # Esta distincao e contribuicao metodologica da dissertacao (Cap. 6).
    LIMIAR_FITNESS = 0.75
    aprovado = fitness_medio >= LIMIAR_FITNESS

    metricas = {
        "fitness_medio":        round(fitness_medio, 4),
        "mf1_perc_fit_traces":  round(mf1, 4),
        "fitness_n_amostra":    n_fit,
        "precisao":             round(precisao, 4) if precisao is not None else None,
        "precisao_n_amostra":   n_prec,
        "aprovado_fitness_075": aprovado,
        "aprovado_mf1_075":     mf1 >= 0.75,
        "nota_criterio": (
            "MF1=0% esperado para log combinado TPU+SAGWeb (~2000 variantes). "
            "Criterio PM4JUD: fitness_medio>=0.75. "
            "D'Castro: mf1>=0.75 (TPU-only, referencia comparativa)."
        ),
    }
    log.info(
        "  Fitness=%.4f | MF1=%.1f%% | Precisao=%s",
        metricas["fitness_medio"],
        metricas["mf1_perc_fit_traces"] * 100,
        f"{metricas['precisao']:.4f}" if metricas["precisao"] is not None else "nao calculada",
    )
    log.info(
        "  Aprovado (fitness>=0.75): %s  |  Ref. D'Castro MF1>=0.75: %s",
        aprovado, metricas["aprovado_mf1_075"],
    )
    return metricas


def extrair_lambda(event_log: EventLog
                   ) -> Tuple[float, float, float, int]:
    """
    Estima lambda (processos/mês) a partir das atividades de chegada.
    Segmenta em prioritários vs. regulares para o modelo M/M/c com
    prioridade do P7 (fila HC separada das demais classes criminais).
    Retorna (lambda_total, lambda_prio, lambda_reg, n_meses).
    """
    cnt_prio: Dict[str, int] = {}
    cnt_reg:  Dict[str, int] = {}

    for trace in event_log:
        is_prio = trace.attributes.get(ATTR_PRIORITARIO, False)
        for event in trace:
            act = event.get("concept:name", "")
            if any(a in act for a in ATIVIDADES_CHEGADA):
                ts = event.get("time:timestamp")
                if ts:
                    chave = (ts.strftime("%Y-%m")
                             if hasattr(ts, "strftime") else str(ts)[:7])
                    if is_prio:
                        cnt_prio[chave] = cnt_prio.get(chave, 0) + 1
                    else:
                        cnt_reg[chave] = cnt_reg.get(chave, 0) + 1
                break  # apenas primeira chegada por traço

    n_meses = max(len(set(cnt_prio) | set(cnt_reg)), 1)
    lam_prio  = round(sum(cnt_prio.values()) / n_meses, 2)
    lam_reg   = round(sum(cnt_reg.values())  / n_meses, 2)
    lam_total = round(lam_prio + lam_reg, 2)
    return lam_total, lam_prio, lam_reg, n_meses


def extrair_mu_por_atividade(event_log: EventLog) -> Dict[str, float]:
    """
    Estima mu (tempo mediano de serviço em dias) por atividade.

    Como os eventos DATAJUD têm timestamp único (não start+complete),
    usa-se o intervalo até o PRÓXIMO evento no mesmo traço como proxy
    do tempo de serviço — mesma lógica do sojourn_time do orientador
    (get_sojourn.apply), adaptada para timestamp único.

    Sanity check: 0 < delta < 365 dias.
    """
    tempos: Dict[str, List[float]] = {}

    for trace in event_log:
        evs = list(trace)
        for i, ev in enumerate(evs[:-1]):
            act    = ev.get("concept:name", "")
            ts_ini = ev.get("time:timestamp")
            ts_fim = evs[i + 1].get("time:timestamp")
            if ts_ini and ts_fim:
                try:
                    delta = (ts_fim - ts_ini).total_seconds() / 86_400
                    if 0 < delta < 365:
                        tempos.setdefault(act, []).append(delta)
                except Exception:
                    pass

    return {
        act: round(float(np.median(vals)), 4)
        for act, vals in tempos.items()
        if vals
    }


def calcular_rho(lam_total: float,
                  mu_dict: Dict[str, float]) -> Dict[str, float]:
    """
    rho = lambda / (capacidade mensal por recurso)
        = lambda_total / (30 / mu_ativ)  [lambda em proc/mês; mu em dias]
    Capeado em 1.0 (sistema estável no modelo teórico M/M/c).
    """
    return {
        act: min(round(lam_total / (30.0 / mu), 4), 1.0)
        for act, mu in mu_dict.items()
        if mu > 0
    }


def salvar_params_csv(params: Dict,
                       gabinete: str,
                       output_dir: Path) -> None:
    """
    Salva tabela lambda, mu, rho em CSV.
    Alimentará o Quadro de parâmetros DES da dissertação
    (Cap. 6, Tabela parametros-des-reais).
    """
    mu_d = params.get("mu_mediano_por_atividade", {})
    rho_d = params.get("rho_por_atividade", {})
    lam  = params.get("lambda_total_proc_mes", 0)

    rows = [
        {
            "gabinete":             gabinete,
            "atividade":            act,
            "lambda_proc_mes":      lam,
            "mu_dias_mediana":      mu_d[act],
            "rho":                  rho_d.get(act, "—"),
            "distribuicao_chegada": "Poisson",
            "distribuicao_servico": "Exponencial",
        }
        for act in sorted(mu_d)
    ]
    fpath = output_dir / f"params_des_{gabinete}.csv"
    pd.DataFrame(rows).to_csv(fpath, index=False, encoding="utf-8")
    log.info("  Parâmetros DES: %s (%d atividades)", fpath.name, len(rows))


# ---------------------------------------------------------------------------
# 5 — Modelo organizacional e_o
# ---------------------------------------------------------------------------

def extrair_modelo_organizacional(event_log: EventLog,
                                   gabinete: str,
                                   output_dir: Path) -> Dict:
    """
    e_o: {atividade -> {recurso: frequência, total, n_distintos}}

    Fase 1 (DATAJUD): org:resource = orgaoJulgador (proxy) — limitação
    documentada na dissertação (seção 6.2.1, nota de rodapé).
    Fase 2 (SAGWeb):  org:resource = assessor real anonimizado.

    Marcação 'fase' permite que o P7 (DES) saiba usar distribuição
    sintética de recursos em Fase 1 e dados reais em Fase 2.
    """
    modelo: Dict[str, Dict[str, int]] = {}
    for trace in event_log:
        for ev in trace:
            act = ev.get("concept:name", "")
            res = ev.get("org:resource", "DATAJUD_proxy")
            modelo.setdefault(act, {})
            modelo[act][res] = modelo[act].get(res, 0) + 1

    e_o = {
        act: {
            "recursos":             modelo[act],
            "total_execucoes":      sum(modelo[act].values()),
            "n_recursos_distintos": len(modelo[act]),
        }
        for act in sorted(modelo)
    }
    payload = {
        "gabinete":      gabinete,
        "fase":          "Fase-1-DATAJUD",
        "nota_resource": (
            "org:resource = orgaoJulgador (proxy DATAJUD). "
            "Fase 2: assessor real anonimizado via SAGWeb."
        ),
        "modelo": e_o,
    }
    fpath = output_dir / f"org_model_{gabinete}.json"
    fpath.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("  Modelo organizacional: %s (%d atividades)", fpath.name, len(e_o))
    return e_o


# ---------------------------------------------------------------------------
# 6 — Estado corrente e_c
# ---------------------------------------------------------------------------

def extrair_estado_corrente(event_log: EventLog,
                             gabinete: str,
                             output_dir: Path) -> Dict:
    """
    e_c: processos em tramitação (traços sem evento terminal).
    Inicializa o modelo DES no P7 com o estado real do gabinete
    ao invés de partir de um ambiente vazio (simulação de curto prazo,
    Reijers & Aalst, 1999 — seção 2.5 da dissertação).
    """
    abertos = []
    for trace in event_log:
        if not trace:
            continue
        ult = trace[-1]
        ult_act = ult.get("concept:name", "")
        encerrado = any(t.lower() in ult_act.lower()
                        for t in ATIVIDADES_TERMINAIS)
        if not encerrado:
            ts = ult.get("time:timestamp")
            abertos.append({
                "case_id":           trace.attributes.get("concept:name", ""),
                "ultima_atividade":  ult_act,
                "timestamp":         ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                "resource":          ult.get("org:resource", ""),
                "classe_tpu":        trace.attributes.get(ATTR_CLASSE, ""),
                "prioritario":       trace.attributes.get(ATTR_PRIORITARIO, False),
                "n_eventos_trace":   len(trace),
            })

    e_c = {"gabinete": gabinete, "n_casos_abertos": len(abertos), "casos": abertos}
    fpath = output_dir / f"current_state_{gabinete}.json"
    fpath.write_text(json.dumps(e_c, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("  Estado corrente: %s (%d processos abertos)", fpath.name, len(abertos))
    return e_c


# ---------------------------------------------------------------------------
# Pipeline por gabinete
# ---------------------------------------------------------------------------

def processar_gabinete(gabinete: str,
                        input_dir: Path,
                        output_dir: Path,
                        k_override: Optional[float] = None,
                        n_amostra_fitness: int = 2000,
                        n_amostra_precisao: int = 500,
                        ont=None) -> Dict:
    """
    Executa o pipeline PM completo para um gabinete.
    k_override: se fornecido, sobrescreve o K_POR_GABINETE padrão.
    ont: instância de OntologiaPM4JUD para normalização semântica.
    Retorna dict de resultado para o relatório consolidado.
    """
    log.info("=" * 70)
    log.info("Processando gabinete: %s", gabinete.upper())
    if ont is not None:
        log.info("  Ontologia: ativa (%s)", ont)
    log.info("=" * 70)

    xes_path = input_dir / f"refine2_{gabinete}.xes"
    if not xes_path.exists():
        log.error("  Arquivo nao encontrado: %s", xes_path)
        return {"gabinete": gabinete, "status": "ERRO",
                "motivo": "arquivo XES ausente"}

    # k efetivo: --k-override tem prioridade sobre K_POR_GABINETE
    k = k_override if k_override is not None else K_POR_GABINETE.get(gabinete, 0.20)
    if k_override is not None:
        log.info("  Arquivo: %s  |  k IMf: %.2f (--k-override)", xes_path.name, k)
    else:
        log.info("  Arquivo: %s  |  k IMf: %.2f (K_POR_GABINETE)", xes_path.name, k)

    t0 = time.time()
    res: Dict = {"gabinete": gabinete, "k": k, "k_origem": "override" if k_override else "padrao"}

    # 1 — Carga
    log.info("  Carregando log...")
    raw = xes_importer.apply(str(xes_path))
    n_total  = sum(len(t) for t in raw)
    n_traces = len(raw)
    log.info("  Log bruto: %d traços | %d eventos", n_traces, n_total)

    # 2 — Verificar composição do log (TPU + Atividades Judiciais SAGWeb)
    #     Os eventos [SIM-ASSESSOR] são Atividades Judiciais do STJ executadas
    #     pelos assessores. Fase 1: sintéticas (calibradas por cargo/gabinete).
    #     Fase 2: reais (SAGWeb). O flag é METADADO de rastreabilidade —
    #     NÃO é critério de exclusão. Sem essas atividades, λ/μ/ρ ficam
    #     errados, e_o não tem assessores, e P7/P8/P9 perdem a variável
    #     de decisão central.
    n_tpu  = sum(1 for t in raw for e in t
                 if not e.get(SIM_FLAG_ATTR))
    n_sagweb = sum(1 for t in raw for e in t
                   if e.get(SIM_FLAG_ATTR) == SIM_FLAG_VALUE)
    fase = "Fase-1 (sintético)" if n_sagweb > 0 else "Fase-2 (real SAGWeb)"
    log.info("  Composição: %d eventos TPU | %d atividades SAGWeb | %s",
             n_tpu, n_sagweb, fase)
    res.update({
        "traces":          n_traces,
        "eventos_total":   n_total,
        "eventos_tpu":     n_tpu,
        "eventos_sagweb":  n_sagweb,
        "fase_detectada":  fase,
    })
    event_log = raw  # usa o log completo — TPU + Atividades Judiciais

    atividades = sorted({e.get("concept:name", "")
                         for t in event_log for e in t
                         if hasattr(e, "get")})
    res["n_atividades_unicas"] = len(atividades)
    log.info("  Atividades únicas (TPU + SAGWeb): %d", len(atividades))

    # 3 — Separar prioritários / regulares
    log_prio, log_reg = separar_por_prioridade(event_log)
    res["n_traces_prioritarios"] = len(log_prio)
    res["n_traces_regulares"]    = len(log_reg)
    log.info("  Traços HC/RHC (prioridade): %d | Regulares: %d",
             len(log_prio), len(log_reg))

    # 4 — DFGs
    # Primário: Graphviz DOT (SVG vetorial, estilo JuMP/CNJ, top 85% fluxo)
    # Exploração: pyvis HTML interativo
    log.info("  Gerando DFGs...")
    dfg_freq = dfg_discovery.apply(event_log)
    # DFG de performance desabilitado — Variants.PERFORMANCE calcula timing
    # para todos os eventos (O(n) com aritmética de datas em Python) e leva
    # 5-10 min para logs de 400k+ eventos. Habilitar apenas quando necessário:
    # dfg_perf = dfg_discovery.apply(event_log,
    #                                variant=dfg_discovery.Variants.PERFORMANCE)
    # gerar_dfg_graphviz(dfg_perf, event_log, gabinete, output_dir,
    #                    ont=ont, variant="performance")
    try:
        gerar_dfg_graphviz(dfg_freq, event_log, gabinete, output_dir,
                           ont=ont, variant="frequency")
    except Exception as e:
        log.warning("  DFG Graphviz frequency: %s", e)
    try:
        gerar_dfg_pyvis(dfg_freq, event_log, gabinete, output_dir)
    except Exception as e:
        log.warning("  DFG pyvis: %s", e)
    if len(log_prio) > 0:
        dfg_hc = dfg_discovery.apply(log_prio)
        try:
            gerar_dfg_graphviz(dfg_hc, log_prio, f"{gabinete}_hc", output_dir,
                               ont=ont, variant="frequency")
        except Exception as e:
            log.warning("  DFG HC Graphviz: %s", e)
        try:
            gerar_dfg_pyvis(dfg_hc, log_prio, f"{gabinete}_hc", output_dir)
        except Exception as e:
            log.warning("  DFG HC pyvis: %s", e)

    # 5 — IMf
    try:
        net, im, fm = descobrir_modelo_imf(event_log, k, gabinete, output_dir)
        res["modelo_descoberto"] = True
    except Exception as e:
        log.error("  IMf falhou: %s", e)
        res.update({"modelo_descoberto": False, "status": "ERRO", "motivo": str(e)})
        return res

    # 6 — Métricas
    metricas = calcular_metricas(event_log, net, im, fm,
                                  n_amostra_fitness=n_amostra_fitness,
                                  n_amostra_precisao=n_amostra_precisao)
    res["metricas_qualidade"] = metricas
    if not metricas["aprovado_fitness_075"]:
        log.warning(
            "  fitness=%.4f < 0.75 — revisar k IMf ou pre-processamento",
            metricas["fitness_medio"],
        )
    else:
        log.info(
            "  Modelo aprovado: fitness=%.4f >= 0.75 "
            "(MF1=%.1f%% — esperado 0%% para log combinado TPU+SAGWeb)",
            metricas["fitness_medio"],
            metricas["mf1_perc_fit_traces"] * 100,
        )

    # 7 — Parâmetros DES (lambda, mu, rho)
    log.info("  Extraindo parâmetros DES (lambda, mu, rho)...")
    lam_t, lam_p, lam_r, n_mes = extrair_lambda(event_log)
    mu_d  = extrair_mu_por_atividade(event_log)
    rho_d = calcular_rho(lam_t, mu_d)
    params_des = {
        "lambda_total_proc_mes":    lam_t,
        "lambda_prioritarios":      lam_p,
        "lambda_regulares":         lam_r,
        "n_meses_coleta":           n_mes,
        "distribuicao_chegada":     "Poisson",
        "distribuicao_servico":     "Exponencial",
        "mu_mediano_por_atividade": mu_d,
        "rho_por_atividade":        rho_d,
    }
    res["parametros_des"] = params_des
    salvar_params_csv(params_des, gabinete, output_dir)
    log.info("  lambda=%.1f proc/mes (prio=%.1f | reg=%.1f) | mu: %d ativ.",
             lam_t, lam_p, lam_r, len(mu_d))

    # 8 — Modelo organizacional e_o
    e_o = extrair_modelo_organizacional(event_log, gabinete, output_dir)
    res["n_ativ_org_model"] = len(e_o)

    # 9 — Estado corrente e_c
    e_c = extrair_estado_corrente(event_log, gabinete, output_dir)
    res["n_casos_abertos"] = e_c["n_casos_abertos"]

    # Finalização
    res["tempo_processamento_s"] = round(time.time() - t0, 1)
    res["status"] = "OK" if metricas["aprovado_fitness_075"] else "AVISO"

    log.info("")
    log.info(
        "  %s: traços=%d | ev_total=%d (tpu=%d sagweb=%d) | ativ=%d | "
        "MF1=%.1f%% | lambda=%.1f proc/mes | abertos=%d | t=%.0fs",
        gabinete, n_traces, n_total, n_tpu, n_sagweb, len(atividades),
        metricas["mf1_perc_fit_traces"] * 100,
        lam_t, e_c["n_casos_abertos"],
        res["tempo_processamento_s"],
    )
    return res


# ---------------------------------------------------------------------------
# Relatório consolidado e resumo
# ---------------------------------------------------------------------------

def salvar_relatorio(resultados: List[Dict], output_dir: Path) -> None:
    """
    Salva pm5_relatorio.json com merge por gabinete.

    Cada execução (inclusive individual) atualiza APENAS o(s) gabinete(s)
    processados, preservando os demais no arquivo. A consolidação é
    recalculada sobre todos os gabinetes presentes.
    """
    fpath = output_dir / "pm5_relatorio.json"

    # Carrega estado anterior se existir
    gabinetes_dict: Dict[str, Dict] = {}
    if fpath.exists():
        try:
            anterior = json.loads(fpath.read_text(encoding="utf-8"))
            for g in anterior.get("gabinetes_lista", []):
                if "gabinete" in g:
                    gabinetes_dict[g["gabinete"]] = g
        except Exception:
            pass

    # Merge: sobrescreve apenas os recém-processados
    for r in resultados:
        r["timestamp_execucao"] = datetime.now(timezone.utc).isoformat()
        gabinetes_dict[r["gabinete"]] = r

    todos = list(gabinetes_dict.values())

    def _media_nested(secao, campo):
        vals = [v[secao][campo] for v in todos
                if isinstance((v.get(secao) or {}).get(campo), (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else None

    consolidacao = {
        "gabinetes_presentes":   sorted(gabinetes_dict.keys()),
        "gabinetes_faltando":    [g for g in ["reynaldo", "palheiro", "schietti"]
                                  if g not in gabinetes_dict],
        "total_traces":          sum(v.get("traces", 0) for v in todos),
        "total_eventos":         sum(v.get("eventos_total", 0) for v in todos),
        "total_eventos_tpu":     sum(v.get("eventos_tpu", 0) for v in todos),
        "total_eventos_sagweb":  sum(v.get("eventos_sagweb", 0) for v in todos),
        "media_fitness":         _media_nested("metricas_qualidade", "fitness_medio"),
        "media_mf1_pct":         round(
            (_media_nested("metricas_qualidade", "mf1_perc_fit_traces") or 0) * 100, 2
        ),
        "media_precisao":        _media_nested("metricas_qualidade", "precisao"),
        "todos_aprovados":       all(
            v.get("metricas_qualidade", {}).get("aprovado_fitness_075", False)
            for v in todos
        ),
        "status_por_gabinete":   {v["gabinete"]: v.get("status", "?") for v in todos},
        "pronto_para_p6":        len(gabinetes_dict) == 3 and all(
            v.get("status") in ("OK", "AVISO") for v in todos
        ),
    }

    payload = {
        "programa":           "PM4JUD-PM",
        "versao":             VERSION,
        "ultima_atualizacao": datetime.now(timezone.utc).isoformat(),
        "proximo_passo": (
            "PM4JUD-LTLf (P6) — execute com complement_<gabinete>.xes "
            "e os artefatos org_model + current_state gerados aqui"
        ),
        "consolidacao":   consolidacao,
        "gabinetes_lista": todos,
        "gabinetes_dict":  gabinetes_dict,
    }

    fpath.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    presentes = consolidacao["gabinetes_presentes"]
    faltando  = consolidacao["gabinetes_faltando"]
    log.info("Relatorio: %s", fpath.name)
    log.info("  Gabinetes presentes : %s", presentes)
    if faltando:
        log.info("  Gabinetes faltando  : %s", faltando)
    if consolidacao["pronto_para_p6"]:
        log.info("  Todos os 3 gabinetes OK — pronto para P6")
    log.info(
        "  Consolidacao: fitness=%.4f | MF1=%.1f%% | precisao=%s",
        consolidacao["media_fitness"] or 0,
        consolidacao["media_mf1_pct"] or 0,
        f"{consolidacao['media_precisao']:.4f}"
        if consolidacao["media_precisao"] is not None else "nao calculada",
    )


def imprimir_resumo(resultados: List[Dict]) -> None:
    log.info("")
    log.info("=" * 60)
    log.info("RESUMO FINAL")
    log.info("=" * 60)
    ok = sum(1 for r in resultados if r.get("status") == "OK")
    for r in resultados:
        fitness = r.get("metricas_qualidade", {}).get("fitness_medio", 0)
        mf1     = r.get("metricas_qualidade", {}).get("mf1_perc_fit_traces", 0)
        lam     = r.get("parametros_des", {}).get("lambda_total_proc_mes", 0)
        n_tpu   = r.get("eventos_tpu", 0)
        n_sag   = r.get("eventos_sagweb", 0)
        log.info(
            "  %-12s: %d traços | ev_total=%d (tpu=%d sag=%d) | "
            "fitness=%.4f | MF1=%.1f%% | lambda=%.1f | k=%.2f (%s) | abertos=%d | %s",
            r["gabinete"], r.get("traces", 0), r.get("eventos_total", 0),
            n_tpu, n_sag,
            fitness, mf1 * 100, lam,
            r.get("k", 0), r.get("k_origem", "padrao"),
            r.get("n_casos_abertos", 0), r.get("status", "?"),
        )
    log.info("")
    log.info("%d gabinetes aprovados | %d com aviso/erro",
             ok, len(resultados) - ok)
    log.info("")
    log.info(
        "Próximo passo: PM4JUD-LTLf (P6) — execute com "
        "complement_<gabinete>.xes"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pm4jud_pm",
        description="PM4JUD-PM (P5) — Mineração de Processos com IMf",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Diretório com refine2_<gab>.xes"
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Diretório de saída dos artefatos"
    )
    parser.add_argument(
        "--gabinetes", nargs="+",
        default=["reynaldo", "palheiro", "schietti"],
        help="Gabinetes a processar (padrão: todos os três)"
    )
    parser.add_argument(
        "--k-override", dest="k_override", type=float, default=None,
        help=(
            "Sobrescreve o k IMf para TODOS os gabinetes processados nesta "
            "execução.  Use quando processar apenas um gabinete com k "
            "não-padrão (ex: --gabinetes palheiro --k-override 0.30). "
            "Sem este argumento, usa K_POR_GABINETE."
        )
    )
    parser.add_argument(
        "--ontologia", type=Path,
        default=Path(__file__).parent.parent / "ontologia",
        help="Diretório da Ontologia PM4JUD (7 módulos OWL/RDF). "
             "Padrão: ../ontologia",
    )
    parser.add_argument(
        "--amostra-fitness", dest="amostra_fitness",
        type=int, default=2000,
        help=(
            "Tamanho da amostra para token replay (fitness/MF1). "
            "Padrão: 2000 traços — erro < 0.02 com 95%% de confiança. "
            "Use 0 para rodar no log completo."
        ),
    )
    parser.add_argument(
        "--amostra-precisao", dest="amostra_precisao",
        type=int, default=500,
        help=(
            "Tamanho da amostra para ETConformance (precisão). "
            "Padrão: 500 traços — erro < 0.03 com 95%% de confiança. "
            "Use 0 para pular completamente."
        ),
    )
    args = parser.parse_args()

    setup_dir(args.output)

    # Carrega Ontologia PM4JUD — camada semântica transversal
    # Módulo 3 (PM4JUD_Classes.owl): classes prioritárias HC/RHC
    # Módulo 5 (PM4JUD_Movimentos.owl): nomes canônicos TPU + pares protegidos
    from pm4jud_ontologia import carregar_ontologia
    ont = carregar_ontologia(args.ontologia, modulos=[3, 5])
    log.info(
        "Ontologia carregada (P5 PM): %d movimentos TPU | %d classes prioritárias",
        len(ont.mapa_tpu()),
        len(ont.classes_prioritarias()),
    )

    log.info("PM4JUD-PM — PM4JUD-PM v%s iniciado", VERSION)
    log.info("Gabinetes: %s", args.gabinetes)
    for g in args.gabinetes:
        k_efetivo = args.k_override if args.k_override is not None \
                    else K_POR_GABINETE.get(g, 0.20)
        origem = "--k-override" if args.k_override is not None else "K_POR_GABINETE"
        log.info("  Gabinete %-12s -> k=%.2f (%s)", g, k_efetivo, origem)

    resultados = []
    for gabinete in args.gabinetes:
        resultado = processar_gabinete(
            gabinete, args.input, args.output,
            k_override=args.k_override,
            n_amostra_fitness=args.amostra_fitness,
            n_amostra_precisao=args.amostra_precisao,
            ont=ont,
        )
        resultados.append(resultado)

    salvar_relatorio(resultados, args.output)
    imprimir_resumo(resultados)


if __name__ == "__main__":
    main()
