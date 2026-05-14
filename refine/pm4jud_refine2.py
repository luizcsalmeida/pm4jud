#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
PM4JUD-REFINE_2  v1.0
================================================================================

Dissertação de Mestrado — PPGIa/PUCPR
Título: PM4JUD — Otimização Multiobjetivo com Mineração de Processos e
        Simulação no Contexto do Fluxo Processual em Gabinetes de Magistrado
Autor:  Luiz Claudio Soares de Almeida
Orient: Prof. Dr. Edson Emilio Scalabrin
Ano:    2026

Descrição
---------
Programa P4 do pipeline PM4JUD. Aplica os três perfis de qualidade de
D'Castro (2020) sobre o log COMPLETO (Movimentos Processuais TPU + Atividades
Judiciais SAGWeb) produzido pelo PM4JUD-COMPLEMENT (P3).

Diferença fundamental em relação ao PM4JUD-REFINE_1 (P2):
  - REFINE_1 opera EXCLUSIVAMENTE sobre Movimentos Processuais TPU (DATAJUD).
  - REFINE_2 opera sobre o log COMPLETO: TPU + atividades internas SAGWeb.
    Os movimentos TPU já chegam refinados do REFINE_1; o foco aqui é a
    qualidade das atividades SAGWeb inseridas pelo COMPLEMENT (Fase 1) ou
    lidas do SAGWeb real (Fase 2).

Os três perfis (D'Castro, 2020):
  Perfil 1 — Canonicalização
    Normaliza rótulos residuais de atividades SAGWeb via CANONICO_INTERNO e
    CANONICAL_MAP. Atividades TPU já vêm canônicas do REFINE_1; Perfil 1
    atua principalmente sobre atividades SAGWeb com nomenclatura legada.

  Perfil 2 — Supressão de atividades infrequentes
    Remove atividades que aparecem em menos de k% dos traços (k calibrado
    por gabinete). Aplicado sobre TODAS as atividades (TPU + SAGWeb).
    Os movimentos TPU raros já foram suprimidos pelo REFINE_1; Perfil 2 aqui
    filtra principalmente atividades SAGWeb geradas para casos excepcionais.

  Perfil 3 — Relabelling de atividades recorrentes
    Dentro de cada traço, atividades SAGWeb que se repetem recebem sufixo
    sequencial (_2, _3, …), preservando a distinção semântica das ocorrências.
    Aplicado APENAS sobre atividades SAGWeb — atividades TPU não são alteradas
    pois sua repetição reflete o fluxo processual real (ex.: múltiplos recursos).

Escopo
------
  Opera sobre o log completo produzido pelo PM4JUD-COMPLEMENT (P3).
  O log de saída é a entrada do PM4JUD-PM (P5).

Pipeline
--------
  Entrada : complement_<gabinete>.xes  (saída do PM4JUD-COMPLEMENT, P3)
  Saída   :
    refine2_<gabinete>.xes              — log completo refinado
    refine2_relatorio.json              — métricas de qualidade antes/depois
    refine2_<gabinete>_suprimidas.csv   — atividades suprimidas pelo Perfil 2
    refine2_<gabinete>_relabelling.csv  — atividades renomeadas pelo Perfil 3

Dependência
-----------
  Requer a saída do PM4JUD-COMPLEMENT (P3).
  Sua saída é entrada obrigatória do PM4JUD-PM (P5).

Referência
----------
D'CASTRO, Raphael José. Abordagem para Pré-processamento de Logs de Eventos
  para Mineração de Processos. 2020. Tese (Doutorado em Ciência da Computação)
  — Universidade Federal de Pernambuco, Recife, 2020.

Repositório: https://github.com/luizcsalmeida/pm4jud/tree/main/refine
================================================================================
"""

# ---------------------------------------------------------------------------
# Imports — biblioteca padrão
# ---------------------------------------------------------------------------
import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Imports — terceiros
# ---------------------------------------------------------------------------
import pandas as pd

try:
    from pm4py.objects.log.obj import EventLog, Trace, Event
    from pm4py.objects.log.importer.xes import importer as xes_importer
    from pm4py.objects.log.exporter.xes import exporter as xes_exporter
except ImportError:
    print("[ERRO] pm4py não encontrado. Execute: pip install pm4py", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Imports — vocabulário canônico compartilhado PM4JUD
# ---------------------------------------------------------------------------
from pm4jud_vocab import (
    CANONICAL_MAP,
    _limpar_tipo_movimento,
)

# K_POR_GABINETE, CANONICO_INTERNO e NAO_RELABELAR foram adicionados ao
# pm4jud_vocab.py na versão 1.1. Se o vocab local for mais antigo, usa
# as definições locais abaixo como fallback.
try:
    from pm4jud_vocab import CANONICO_INTERNO, NAO_RELABELAR
except ImportError:
    CANONICO_INTERNO: Dict[str, str] = {
        "Escaninho: Em analise":                         "Em analise",
        "Escaninho: Recebido":                           "Recebido pelo assessor",
        "Escaninho: Aguardando julgamento":              "Aguardando sessao",
        "Assinatura: Ministro":                          "Assinatura de documento pelo ministro",
        "Deslocamento: Para turma":                      "Deslocamento: Gabinete para Turma",
        "Criacao de documento: Relatorio conclusivo":    "Criacao de documento: RELATORIO E VOTO",
        "Criacao de documento: Minuta de voto":          "Criacao de documento: RELATORIO E VOTO",
        "Alteracao de documento: Relatorio conclusivo":  "Alteracao de documento: RELATORIO E VOTO",
        "Alteracao de documento: Minuta de voto":        "Alteracao de documento: RELATORIO E VOTO",
    }
    NAO_RELABELAR: Set[str] = {
        "Julgado", "Arquivado", "Publicacao de documento no DJe",
        "Assinatura de documento pelo ministro",
        "Certidao de Julgamento",
        "Criacao de documento: EMENTA ACORDAO",
        "Criacao de documento: DESPACHO DECISAO",
        "Envio coordenadoria: DESPACHO DECISAO",
        "Envio coordenadoria: RELATORIO E VOTO",
        "Envio coordenadoria: EMENTA ACORDAO",
    }

# Limiar k calibrado empiricamente por gabinete (mai/2026).
# Valores derivados da maximização do MF1 no corpus DATAJUD 2024
# (32.031 processos — Rey=11.395, Pal=10.148, Sch=10.488).
# Definido localmente para que o REFINE_2 seja auto-suficiente —
# não depende da versão do pm4jud_vocab instalada no ambiente.
# Fonte: pm4jud_diagnostico_tpu.py — maximização iterativa do MF1.
K_POR_GABINETE: Dict[str, float] = {
    "reynaldo": 0.20,   # MF1=92,1%
    "palheiro": 0.30,   # MF1=77,5%
    "schietti": 0.25,   # MF1=81,9%
}

# ==============================================================================
# CONSTANTES
# ==============================================================================

GABINETES_PADRAO = ["reynaldo", "palheiro", "schietti"]
LIMIAR_FREQUENCIA_PADRAO: float = 0.20   # fallback global (sem calibração)

# Atributos XES
ATTR_ACT    = "concept:name"
ATTR_TS     = "time:timestamp"
ATTR_SIM    = "pm4jud:sim_flag"
ATTR_GRP    = "pm4jud:grupo_sagweb"
ATTR_ORIGEM = "pm4jud:refine2_perfis"

# Prefixo que identifica atividades SAGWeb simuladas (Fase 1)
SIM_PREFIX = "[SIM"

# ==============================================================================
# UTILITÁRIOS
# ==============================================================================

def _is_sagweb(ev: Event) -> bool:
    """
    Retorna True se o evento é uma atividade SAGWeb — seja simulada (Fase 1)
    ou real (Fase 2, sem sim_flag mas com pm4jud:grupo_sagweb definido).
    """
    sim = str(ev.get(ATTR_SIM, ""))
    grp = ev.get(ATTR_GRP)
    return sim.startswith(SIM_PREFIX) or (grp is not None and grp != "")


def _freq_atividades(log: EventLog) -> Dict[str, float]:
    """
    Calcula a frequência relativa de cada atividade no log.

    Definição de frequência (D'Castro, 2020)
    -----------------------------------------
    freq(A) = |{traços que contêm A pelo menos uma vez}| / |total de traços|

    Esta é uma medida BASEADA EM TRAÇOS, não em eventos.
    Se uma atividade aparece 50 vezes em 10 traços, sua frequência é
    10/N (quantos traços a contêm), não 50/N (quantos eventos existem).

    Por que frequência por traço e não por evento?
    -----------------------------------------------
    O Perfil 2 visa remover atividades RARAS NO CONTEXTO DO PROCESSO —
    aquelas que ocorrem em poucos tipos de casos. Uma atividade que aparece
    1000 vezes mas apenas em 5 % dos casos é igualmente especializada que
    uma que aparece 1 vez em 5 % dos casos. A frequência por traço captura
    essa dimensão de "universalidade" da atividade no fluxo processual.

    Complexidade: O(N × max_eventos_por_traço) com N = número de traços.
    O uso de um set por traço garante que cada atividade seja contada
    no máximo uma vez por traço, mesmo que ocorra múltiplas vezes.

    Parameters
    ----------
    log : EventLog
        Log completo (TPU + SAGWeb) pós-canonicalização (pós-Perfil 1).

    Returns
    -------
    Dict[str, float]
        {concept_name: frequência_relativa} para todas as atividades únicas.
    """
    n = len(log)
    if n == 0:
        return {}
    contagem: Dict[str, int] = defaultdict(int)
    for trace in log:
        vistos: Set[str] = set()
        for ev in trace:
            nome = ev.get(ATTR_ACT, "")
            if nome and nome not in vistos:
                contagem[nome] += 1
                vistos.add(nome)
    return {k: v / n for k, v in contagem.items()}


def log_setup(out: Path, nome: str = "PM4JUD-REFINE_2") -> logging.Logger:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = out / f"pm4jud_refine2_{ts}.log"
    fmt = "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_path), encoding="utf-8"),
        ],
        force=True,
    )
    return logging.getLogger(nome)


# ==============================================================================
# PROCESSADOR PRINCIPAL
# ==============================================================================

class ProcessadorRefine2:
    """
    Aplica os três perfis D'Castro sobre o log completo (TPU + SAGWeb).

    Posição no pipeline
    -------------------
    Entrada : complement_<gab>.xes — log enriquecido pelo P3 COMPLEMENT.
              Contém Movimentos Processuais TPU (já refinados pelo P2) E
              Atividades Judiciais SAGWeb (geradas/lidas pelo P3).
    Saída   : refine2_<gab>.xes — pronto para descoberta de modelos no P5.

    Por que aplicar D'Castro novamente?
    ------------------------------------
    O P2 REFINE_1 aplicou D'Castro apenas sobre os movimentos TPU.
    O P3 COMPLEMENT injetou atividades SAGWeb sem tratamento de qualidade.
    O P4 REFINE_2 fecha o ciclo: aplica os mesmos critérios de qualidade
    sobre o log COMPLETO, garantindo que as atividades SAGWeb atendam aos
    mesmos padrões de frequência e canonicidade que os movimentos TPU.

    Os três perfis e suas adaptações para o log completo
    -----------------------------------------------------
    Perfil 1 — Canonicalização (apenas SAGWeb)
        Atividades TPU chegam canônicas do REFINE_1; não são modificadas.
        Atividades SAGWeb podem ter nomes legados (ex.: "Relatorio conclusivo"
        → "RELATORIO E VOTO") — o CANONICO_INTERNO resolve essas variantes.
        Em Fase 1 (sintético) o Perfil 1 tipicamente não dispara porque o
        COMPLEMENT já gera nomes canônicos. Em Fase 2 (real) pode disparar
        para atividades exportadas com nomenclatura antiga do SAGWeb.

    Perfil 2 — Supressão de infrequentes (TPU + SAGWeb)
        Frequência = fração de TRAÇOS em que a atividade aparece ao menos
        uma vez (não contagem de eventos). Atividades abaixo de k são
        removidas. O efeito principal é sobre atividades SAGWeb de casos
        excepcionais (ex.: G2c Exclusão de documento, G2e Jurisprudência).
        Atividades TPU raras já foram filtradas pelo REFINE_1 — aqui o
        Perfil 2 reequilibra o alfabeto com a nova composição TPU + SAGWeb.

    Perfil 3 — Relabelling de recorrentes (apenas SAGWeb)
        Atividades TPU NÃO são relabeladas — sua repetição é semanticamente
        significativa (múltiplos recursos, redistribuições, declarações de
        voto). Atividades SAGWeb que se repetem DENTRO de um mesmo traço
        recebem sufixo sequencial (_2, _3...), distinguindo sessões de
        edição distintas. Ex.: "Alteracao de documento: DESPACHO DECISAO_3"
        = terceira edição do mesmo despacho no mesmo processo.
    """
    """
    Aplica os três perfis D'Castro sobre o log completo (TPU + SAGWeb).
    Produz o log refinado e artefatos de rastreabilidade por gabinete.
    """

    def __init__(self, output_dir: Path, logger: logging.Logger):
        self.out = output_dir
        self.log = logger

    # ------------------------------------------------------------------
    # Perfil 1 — Canonicalização de rótulos SAGWeb residuais
    # ------------------------------------------------------------------

    def _perfil1_canonicalizar(
        self, log: EventLog
    ) -> Tuple[EventLog, int]:
        """
        Perfil 1: Canonicalização de rótulos de atividades SAGWeb residuais.

        Descrição do perfil (D'Castro, 2020 — Seção 4.1)
        --------------------------------------------------
        O Perfil 1 identifica e unifica atividades com nomes semanticamente
        equivalentes mas textualmente distintos. Na implementação original
        de D'Castro, isso envolve similaridade TF-IDF. Aqui, usamos lookup
        direto via dicionários porque:

          1. As atividades SAGWeb têm vocabulário controlado — não há variação
             livre de texto como em logs industriais.
          2. Os poucos casos de variação são conhecidos (legado v3 → v4) e
             estão mapeados em CANONICO_INTERNO.
          3. Lookup O(1) é mais eficiente e determinístico que TF-IDF.

        Prioridade de canonicalização
        ------------------------------
        1º CANONICO_INTERNO — normaliza nomes SAGWeb legados (ex.: "Relatorio
           conclusivo" → "RELATORIO E VOTO"). Tabelado em pm4jud_vocab.py.
        2º _limpar_tipo_movimento + CANONICAL_MAP — remove templates #{...}
           residuais e aplica o vocabulário canônico do ETL.

        Atividades TPU não são tocadas: chegam canônicas do REFINE_1 e seu
        rótulo não deve ser alterado (são indivíduos OWL da ontologia).

        Quando dispara?
        ---------------
        Em Fase 1 (sintético): raramente — o COMPLEMENT já gera nomes v4.
        Em Fase 2 (real SAGWeb): pode disparar para atividades exportadas
        com nomenclatura anterior à atualização da ontologia.

        Returns
        -------
        Tuple[EventLog, int]
            (log_com_nomes_canonicos, número_de_eventos_canonicalizados)
        """
        saida = EventLog()
        saida.attributes.update(log.attributes)
        n_canon = 0

        for trace in log:
            t2 = Trace()
            t2.attributes.update(trace.attributes)
            for ev in trace:
                if _is_sagweb(ev):
                    nome = ev.get(ATTR_ACT, "")
                    # Tenta CANONICO_INTERNO primeiro (nomes SAGWeb)
                    novo = CANONICO_INTERNO.get(nome)
                    if novo is None:
                        # Fallback: tenta limpeza de template + CANONICAL_MAP
                        limpo = _limpar_tipo_movimento(nome)
                        novo = CANONICAL_MAP.get(limpo, limpo)
                    if novo != nome:
                        ev[ATTR_ACT] = novo
                        n_canon += 1
                t2.append(ev)
            saida.append(t2)

        return saida, n_canon

    # ------------------------------------------------------------------
    # Perfil 2 — Supressão de atividades infrequentes
    # ------------------------------------------------------------------

    def _perfil2_suprimir(
        self, log: EventLog, k: float
    ) -> Tuple[EventLog, int, Dict[str, float]]:
        """
        Perfil 2: Supressão de atividades infrequentes.

        Descrição do perfil (D'Castro, 2020 — Seção 4.2)
        --------------------------------------------------
        Remove do log todas as atividades cuja frequência relativa (fração
        de traços em que aparecem) seja inferior ao limiar k.

        Fundamentação
        --------------
        Atividades infrequentes indicam variantes processuais excepcionais
        (erros, casos atípicos, eventos isolados). Se incluídas no modelo
        de processo, geram ramificações com suporte estatístico insuficiente,
        degradando fitness e precisão do modelo descoberto pelo P5.

        Limiar k calibrado por gabinete
        --------------------------------
        O valor de k é específico por gabinete, derivado da maximização do
        MF1 (métrica combinada de fitness e precisão) sobre o corpus DATAJUD:
          Reynaldo: k=0.20  (MF1=92,1%)
          Palheiro: k=0.30  (MF1=77,5%)
          Schietti: k=0.25  (MF1=81,9%)

        Atividades TPU no REFINE_2
        ---------------------------
        Os movimentos TPU já passaram pelo mesmo critério no REFINE_1 (P2).
        No REFINE_2, o Perfil 2 reequilibra o alfabeto com a nova composição
        TPU + SAGWeb — pode remover atividades SAGWeb excepcionais (G2c
        Exclusão, G2e Jurisprudência) que aparecem em < k traços.

        Efeito no alfabeto
        ------------------
        Reynaldo corpus: 132 atividades únicas → 53 após P2 (−59,9%).
        O IMf do P5 opera melhor com 50–80 atividades; spaghetti inicia
        acima de 100.

        Parameters
        ----------
        log : EventLog
            Log pós-Perfil 1 (nomes já canonicalizados).
        k : float
            Limiar de frequência mínima [0, 1]. Calibrado por gabinete.

        Returns
        -------
        Tuple[EventLog, int, Dict[str, float]]
            (log_filtrado, n_eventos_removidos, {atividade: freq_removida})
        """
        freq = _freq_atividades(log)
        suprimidas = {nome: f for nome, f in freq.items() if f < k}
        self.log.info(
            "  Perfil 2: k=%.2f | %d atividades suprimidas de %d únicas",
            k, len(suprimidas), len(freq),
        )

        saida = EventLog()
        saida.attributes.update(log.attributes)
        n_sup = 0

        for trace in log:
            t2 = Trace()
            t2.attributes.update(trace.attributes)
            for ev in trace:
                nome = ev.get(ATTR_ACT, "")
                if nome in suprimidas:
                    n_sup += 1
                    continue
                t2.append(ev)
            if t2:
                saida.append(t2)

        return saida, n_sup, suprimidas

    # ------------------------------------------------------------------
    # Perfil 3 — Relabelling de atividades SAGWeb recorrentes
    # ------------------------------------------------------------------

    def _perfil3_relabelar(
        self, log: EventLog
    ) -> Tuple[EventLog, int, Dict[str, int]]:
        """
        Dentro de cada traço, atividades SAGWeb que se repetem recebem sufixo
        sequencial (_2, _3, …). Atividades TPU NÃO são modificadas — sua
        repetição é semanticamente significativa (múltiplos recursos, julgamentos).

        Returns
        -------
        (log_resultado, n_relabelados, contagem_por_atividade)
        """
        saida = EventLog()
        saida.attributes.update(log.attributes)
        n_rel = 0
        total_por_ativ: Dict[str, int] = defaultdict(int)

        for trace in log:
            t2 = Trace()
            t2.attributes.update(trace.attributes)
            contagem: Dict[str, int] = defaultdict(int)

            for ev in trace:
                nome = ev.get(ATTR_ACT, "")
                if _is_sagweb(ev) and nome and nome not in NAO_RELABELAR:
                    contagem[nome] += 1
                    if contagem[nome] > 1:
                        novo = f"{nome}_{contagem[nome]}"
                        ev[ATTR_ACT] = novo
                        n_rel += 1
                        total_por_ativ[nome] += 1
                t2.append(ev)

            saida.append(t2)

        return saida, n_rel, dict(total_por_ativ)

    # ------------------------------------------------------------------
    # Orquestração por gabinete
    # ------------------------------------------------------------------

    def processar_gabinete(
        self,
        chave: str,
        xes_path: Path,
        limiar_k: Optional[float] = None,
    ) -> Optional[EventLog]:
        """
        Executa os três perfis de refinamento sobre o log completo do gabinete.

        Parameters
        ----------
        chave : str
            Identificador do gabinete (ex.: "reynaldo").
        xes_path : Path
            Caminho do arquivo complement_<chave>.xes (saída do P3).
        limiar_k : float, opcional
            Limiar de frequência para o Perfil 2. Se None, usa K_POR_GABINETE.
        """
        limiar_efetivo = limiar_k if limiar_k is not None else K_POR_GABINETE.get(
            chave, LIMIAR_FREQUENCIA_PADRAO
        )

        self.log.info("=" * 70)
        self.log.info("Processando gabinete: %s", chave.upper())
        self.log.info("Arquivo de entrada: %s", xes_path)
        self.log.info("k efetivo (Perfil 2): %.2f", limiar_efetivo)
        self.log.info("=" * 70)

        if not xes_path.exists():
            self.log.error("Arquivo não encontrado: %s", xes_path)
            return None

        # Carrega log completo (TPU + SAGWeb)
        self.log.info("  Carregando log completo...")
        log_orig = xes_importer.apply(str(xes_path))
        n_traces_orig  = len(log_orig)
        n_eventos_orig = sum(len(t) for t in log_orig)
        n_ativ_orig    = len({ev.get(ATTR_ACT,"") for t in log_orig for ev in t})
        n_sagweb_orig  = sum(1 for t in log_orig for ev in t if _is_sagweb(ev))
        self.log.info(
            "  Log carregado: %d traços | %d eventos | %d atividades únicas",
            n_traces_orig, n_eventos_orig, n_ativ_orig,
        )
        self.log.info(
            "  Composição: %d eventos TPU | %d eventos SAGWeb",
            n_eventos_orig - n_sagweb_orig, n_sagweb_orig,
        )

        # ── Perfil 1 ─────────────────────────────────────────────────
        self.log.info("  Perfil 1 — Canonicalização...")
        log_p1, n_canon = self._perfil1_canonicalizar(log_orig)
        self.log.info("  Perfil 1: %d atividades SAGWeb canonicalizadas", n_canon)

        # ── Perfil 2 ─────────────────────────────────────────────────
        self.log.info("  Perfil 2 — Supressão de infrequentes...")
        log_p2, n_sup, suprimidas = self._perfil2_suprimir(log_p1, limiar_efetivo)
        self.log.info("  Perfil 2: %d eventos suprimidos", n_sup)

        # ── Perfil 3 ─────────────────────────────────────────────────
        self.log.info("  Perfil 3 — Relabelling de SAGWeb recorrentes...")
        log_p3, n_rel, rel_contagem = self._perfil3_relabelar(log_p2)
        self.log.info("  Perfil 3: %d eventos relabelados", n_rel)

        # Marca rastreabilidade
        for trace in log_p3:
            for ev in trace:
                ev[ATTR_ORIGEM] = ATTR_ORIGEM

        # ── Métricas finais ───────────────────────────────────────────
        n_eventos_fin = sum(len(t) for t in log_p3)
        n_ativ_fin    = len({ev.get(ATTR_ACT,"") for t in log_p3 for ev in t})
        taxa_sup      = round(n_sup / max(n_eventos_orig, 1) * 100, 2)
        taxa_red_ativ = round((1 - n_ativ_fin / max(n_ativ_orig, 1)) * 100, 2)

        self.log.info(
            "\n  %s: traços=%d | eventos: %d→%d (−%.1f%%) | "
            "ativ. únicas: %d→%d (−%.1f%%)\n"
            "  P1=%d canon | P2=%d suprim (%.1f%%) | P3=%d relabel",
            chave, len(log_p3),
            n_eventos_orig, n_eventos_fin, taxa_sup,
            n_ativ_orig, n_ativ_fin, taxa_red_ativ,
            n_canon, n_sup, taxa_sup, n_rel,
        )

        # ── Exporta XES ───────────────────────────────────────────────
        xes_out = self.out / f"refine2_{chave}.xes"
        xes_exporter.apply(log_p3, str(xes_out))
        self.log.info("  XES: %s", xes_out.name)

        # ── CSVs de rastreabilidade ───────────────────────────────────
        # Suprimidas
        df_sup = pd.DataFrame([
            {"atividade": nome, "frequencia_relativa": round(freq, 4),
             "limiar_k": limiar_efetivo}
            for nome, freq in sorted(suprimidas.items(), key=lambda x: x[1])
        ])
        df_sup.to_csv(
            self.out / f"refine2_{chave}_suprimidas.csv",
            index=False, encoding="utf-8",
        )

        # Relabelling
        df_rel = pd.DataFrame([
            {"atividade_original": nome, "n_ocorrencias_relabeladas": cnt}
            for nome, cnt in sorted(rel_contagem.items(), key=lambda x: -x[1])
        ])
        df_rel.to_csv(
            self.out / f"refine2_{chave}_relabelling.csv",
            index=False, encoding="utf-8",
        )

        # Retorna estatísticas para o relatório JSON
        log_p3._stats = {
            "gabinete":             chave,
            "n_traces":             len(log_p3),
            "n_eventos_originais":  n_eventos_orig,
            "n_eventos_finais":     n_eventos_fin,
            "n_sagweb_originais":   n_sagweb_orig,
            "n_atividades_orig":    n_ativ_orig,
            "n_atividades_finais":  n_ativ_fin,
            "taxa_supressao_pct":   taxa_sup,
            "reducao_atividades_pct": taxa_red_ativ,
            "p1_canonicalizados":   n_canon,
            "p2_suprimidos":        n_sup,
            "p2_atividades_removidas": len(suprimidas),
            "p3_relabelados":       n_rel,
            "limiar_k":             limiar_efetivo,
            "status":               "OK",
        }
        return log_p3


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        prog="pm4jud_refine2",
        description=(
            "PM4JUD-REFINE_2 v1.0 — Refinamento do log completo (TPU + SAGWeb) "
            "segundo os três perfis de qualidade (D'Castro, 2020)."
        ),
        epilog=(
            "Exemplos:\n"
            "  python pm4jud_refine2.py --input ./output --output ./output\n"
            "  python pm4jud_refine2.py --input ./output --gabinetes reynaldo\n"
            "  python pm4jud_refine2.py --input ./output --limiar-k 0.25"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input",  type=Path, required=True,
        help="Diretório com os arquivos complement_<gabinete>.xes (saída do P3).",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Diretório de saída. Padrão: mesmo que --input.",
    )
    p.add_argument(
        "--gabinetes", nargs="+", default=GABINETES_PADRAO,
        metavar="GAB",
        help=f"Gabinetes a processar. Padrão: {GABINETES_PADRAO}",
    )
    p.add_argument(
        "--limiar-k", type=float, default=None,
        dest="limiar_k",
        help=(
            "Limiar de frequência para o Perfil 2 (sobrescreve K_POR_GABINETE). "
            "Padrão: usa calibração por gabinete "
            "(reynaldo=0.20, palheiro=0.30, schietti=0.25)."
        ),
    )
    p.add_argument(
        "--ontologia", type=Path,
        default=Path(__file__).parent.parent / "ontologia",
        help="Diretório da Ontologia PM4JUD (7 módulos OWL/RDF). "
             "Padrão: ../ontologia",
    )

    args = p.parse_args()
    out = args.output or args.input
    out.mkdir(parents=True, exist_ok=True)

    logger = log_setup(out)
    logger.info("PM4JUD-REFINE_2 v1.0 iniciado")
    logger.info("Gabinetes: %s", args.gabinetes)

    # Carrega Ontologia PM4JUD — camada semântica transversal (Módulos 2 e 4)
    # Módulo 4: canonicalização semântica TPU + SAGWeb
    # Módulo 2: detecção de classes prioritárias
    from pm4jud_ontologia import carregar_ontologia
    ont = carregar_ontologia(args.ontologia, modulos=[3, 5])
    logger.info(
        "Ontologia carregada (P4 REFINE_2): %d movimentos TPU | %d classes prioritárias",
        len(ont.mapa_tpu()),
        len(ont.classes_prioritarias()),
    )

    processador = ProcessadorRefine2(output_dir=out, logger=logger)

    relatorio = {
        "programa":   "PM4JUD-REFINE_2",
        "versao":     "1.0",
        "gerado_em":  datetime.now(tz=timezone.utc).isoformat(),
        "gabinetes":  [],
    }

    n_ok = n_erro = 0

    for chave in args.gabinetes:
        k_gab = K_POR_GABINETE.get(chave, LIMIAR_FREQUENCIA_PADRAO)
        if args.limiar_k is not None:
            k_gab = args.limiar_k
        logger.info("  Gabinete %-12s → k=%.2f", chave, k_gab)

        # Candidatos de entrada (em ordem de preferência)
        candidatos = [
            args.input / f"complement_{chave}.xes",   # P3 COMPLEMENT — preferencial
        ]
        xes_path = next((c for c in candidatos if c.exists()), None)

        if xes_path is None:
            logger.error(
                "Nenhum arquivo encontrado para '%s'. "
                "Execute o PM4JUD-COMPLEMENT (P3) primeiro.", chave
            )
            relatorio["gabinetes"].append({
                "gabinete": chave, "status": "ARQUIVO_NAO_ENCONTRADO"
            })
            n_erro += 1
            continue

        try:
            log_refinado = processador.processar_gabinete(
                chave, xes_path, limiar_k=k_gab
            )
            if log_refinado is not None:
                stats = getattr(log_refinado, "_stats", {"gabinete": chave, "status": "OK"})
                relatorio["gabinetes"].append(stats)
                n_ok += 1
            else:
                relatorio["gabinetes"].append({"gabinete": chave, "status": "ERRO"})
                n_erro += 1
        except Exception as exc:
            logger.error("Erro '%s': %s", chave, exc, exc_info=True)
            relatorio["gabinetes"].append({"gabinete": chave, "status": f"ERRO — {exc}"})
            n_erro += 1

    # Salva relatório JSON com merge por gabinete
    from pm4jud_vocab import salvar_relatorio_com_merge
    salvar_relatorio_com_merge(
        fpath=out / "refine2_relatorio.json",
        programa="PM4JUD-REFINE_2",
        versao="1.0",
        novos_resultados=relatorio["gabinetes"],
        campos_consolidacao={
            "n_eventos_entrada":  "total_eventos_entrada",
            "n_eventos_saida":    "total_eventos_saida",
            "n_traces":           "total_traces",
            "n_ativ_antes":       "total_ativ_antes",
            "n_ativ_depois":      "total_ativ_depois",
        },
    )
    logger.info("  JSON: %s", rel_path.name)

    # Resumo final
    logger.info("\n%s\nRESUMO FINAL\n%s", "=" * 60, "=" * 60)
    for g in relatorio["gabinetes"]:
        if g.get("status") == "OK":
            logger.info(
                "  %-12s: %d traços | %d→%d eventos | "
                "P1=%d P2=%d P3=%d | k=%.2f",
                g["gabinete"], g["n_traces"],
                g["n_eventos_originais"], g["n_eventos_finais"],
                g["p1_canonicalizados"], g["p2_suprimidos"], g["p3_relabelados"],
                g["limiar_k"],
            )
        else:
            logger.error("  %-12s: %s", g["gabinete"], g["status"])

    logger.info(
        "\n%s aprovados | %s com erro",
        f"{n_ok} gabinete{'s' if n_ok != 1 else ''}",
        f"{n_erro} gabinete{'s' if n_erro != 1 else ''}",
    )
    logger.info(
        "Próximo passo: PM4JUD-PM (P5) — "
        "execute com os arquivos refine2_<gabinete>.xes como entrada."
    )


if __name__ == "__main__":
    main()
