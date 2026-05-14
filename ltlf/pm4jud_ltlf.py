#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
PM4JUD-LTLf  v1.0
================================================================================

Dissertação de Mestrado — PPGIa/PUCPR
Título: PM4JUD — Otimização Multiobjetivo com Mineração de Processos e
        Simulação no Contexto do Fluxo Processual em Gabinetes de Magistrado
Autor:  Luiz Claudio Soares de Almeida
Orient: Prof. Dr. Edson Emilio Scalabrin
Ano:    2026

Descrição
---------
Aplica os 16 constraints C1–C16 (PM4JUD.owl Módulo 7 — RISTJ) usando a
API nativa de conformance checking Declare do PM4Py.

Architetura
-----------
Ontologia (Módulo 7)
  └── codigoRegra, padraoLTLf, temAtividadeAntecedente/Consequente
          │
          ▼
  build_declare_model()        ← converte constraints OWL → formato PM4Py Declare
          │                      {template: {(atividade_A, atividade_B): {}}}
          ▼
  pm4py.algo.conformance       ← conformance checking nativo
  .declare.algorithm.apply()     retorna dev_fitness por trace
          │
          ▼
  κ = média dev_fitness        ← taxa de conformidade regimental
  η = Metas CNJ 1/2/4          ← nível de log (calcular_eta)

Templates Declare suportados (alinhados ao padraoLTLf da ontologia)
-------------------------------------------------------------------
  existence              → activity must occur at least once
  absence                → activity must not occur
  response               → if A occurs, B must eventually follow
  precedence             → B can only occur if A occurred before
  responded_existence    → if A occurs, B must occur (before or after)
  chainresponse          → if A occurs, B must immediately follow
  chainprecedence        → B can only occur if A immediately preceded
  coexistence            → A and B must both occur or neither
  init                   → first activity must be A

Constraints temporais (response_with_deadline)
----------------------------------------------
PM4Py não implementa bounded response nativamente. Para C1/C10/C11/C13,
usa-se RESPONSE como verificação estrutural + verificador de prazo em Python.
O padraoLTLf "response_with_deadline" é mapeado para um verificador híbrido.

Metas CNJ (C7/C8/C9 — aggregate_count / response_with_deadline log-level)
--------------------------------------------------------------------------
Não são per-trace — calculadas em calcular_eta() sobre o log completo.

Entradas (output/ do P4/P5)
----------------------------
  refine2_<gab>.xes          Log completo TPU + SAGWeb (do P4)
  current_state_<gab>.json   Estado corrente (do P5)
  params_des_<gab>.csv       Parâmetros DES (do P5)
  --ontologia  <dir>         Diretório com PM4JUD_*.owl

Saídas (output/)
----------------
  ltlf_<gab>.json            Resultado por constraint e por trace
  p6_relatorio.json          Resumo consolidado (κ e η por gabinete)

Pipeline
--------
  P1→P2→P3→P4→P5→[P6 LTLf]→P7 DES→P8 OPT→P9 STAT

Repositório: https://github.com/luizcsalmeida/pm4jud/tree/main/ltlf
================================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np

# ---------------------------------------------------------------------------
# PM4Py
# ---------------------------------------------------------------------------
try:
    from pm4py.objects.log.obj import EventLog, Trace
    from pm4py.objects.log.importer.xes import importer as xes_importer
    from pm4py.algo.conformance.declare.algorithm import apply as declare_check
    from pm4py.algo.discovery.declare import templates as DT
except ImportError as exc:
    print(f"[ERRO] {exc}\n       pip install pm4py==2.7.8.3", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Ontologia PM4JUD
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent.parent
if str(_HERE / "etl") not in sys.path:
    sys.path.insert(0, str(_HERE / "etl"))

try:
    from pm4jud_ontologia import OntologiaPM4JUD, NS_ASSUNTOS
    _ONT_OK = True
except ImportError:
    _ONT_OK = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] PM4JUD-LTLf — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("PM4JUD-LTLf")
VERSION = "2.0"   # v2: usa PM4Py Declare nativo

# ===========================================================================
# SEÇÃO 1 — Constantes do domínio STJ
# ===========================================================================

ATTR_PRIO    = "pm4jud:prioritario"
ATTR_SIM     = "pm4jud:sim_flag"
ATTR_TPU_COD = "pm4jud:tpu_code"
ATTR_ASSUNTOS = "pm4jud:assuntos"   # formato: "código:Nome; código2:Nome2; ..."
ATTR_CLASSE  = "pm4jud:classe"

# Prazos regimentais (dias corridos)
PRAZO_MONO    = 10   # Art. 110-I
PRAZO_ACORDAO = 30   # Art. 110-III / Art. 179

# Conjuntos de atividades TPU — para calcular_eta (Metas CNJ)
_DIST   = {"Distribuído", "Distribuído por sorteio eletrônico",
           "Recebidos os autos pelo gabinete do relator"}
_SESSAO = {"Fase externa: Julgamento", "Incluído em pauta",
           "Julgamento iniciado", "Incluído em Pauta de Julgamento"}
_ACORDAO= {"Publicado acórdão no DJEN", "Publicação de acórdão",
           "Disponibilização no Diário da Justiça Eletrônico"}
_TRAN   = {"Trânsito em julgado"}
_BAIXA  = {"Baixa Definitiva", "Baixado definitivamente", "Baixado"}
_CONCL  = {"Conclusão para decisão", "Conclusão para despacho", "Conclusão"}

# Palavras-chave de resultado HC (para Meta CNJ 4a fallback)
_RESULT_HC_PAL  = {"concedido", "denegado", "não conhecido", "prejudicado"}
_RESULT_HC_CODS = {"443", "451", "12458", "12475"}

# Cache hierarquia assuntos
_CATS: Dict[str, Set[int]] = {}

# ===========================================================================
# SEÇÃO 2 — Mapeamento padraoLTLf → template PM4Py Declare
# ===========================================================================

# Padrões da ontologia que mapeiam diretamente a templates PM4Py
_PADRAO_TO_TEMPLATE: Dict[str, str] = {
    "existence":          DT.EXISTENCE,
    "absence":            DT.ABSENCE,
    "response":           DT.RESPONSE,
    "precedence":         DT.PRECEDENCE,
    "responded_existence":DT.RESPONDED_EXISTENCE,
    "chain_response":     DT.CHAINRESPONSE,
    "chainresponse":      DT.CHAINRESPONSE,
    "chain_precedence":   DT.CHAINPRECEDENCE,
    "chainprecedence":    DT.CHAINPRECEDENCE,
    "coexistence":        DT.COEXISTENCE,
    "init":               DT.INIT,
}

# Padrões que NÃO têm suporte nativo no PM4Py — tratados separadamente
_PADRAO_TEMPORAL    = "response_with_deadline"   # C1/C10/C11/C13
_PADRAO_AGREGADO    = "aggregate_count"           # C7/C8/C9 — log-level
_PADRAO_RECURSO     = "cond_resource"             # C3 — recurso especializado

# IDs das constraints log-level (Metas CNJ) — excluídas do loop por-trace
IDS_LOG_LEVEL = {"C7", "C8", "C9"}

# ===========================================================================
# SEÇÃO 3 — Carregamento da ontologia e construção do modelo Declare
# ===========================================================================

def carregar_ontologia(ont_dir: Optional[Path]) -> Optional["OntologiaPM4JUD"]:
    if not _ONT_OK or ont_dir is None or not ont_dir.exists():
        log.warning("  Ontologia: indisponível — usando fallback embutido")
        return None
    try:
        ont = OntologiaPM4JUD(ont_dir).carregar([7])
        log.info("  Ontologia Módulo 7: PM4JUD.owl carregada")
        return ont
    except Exception as exc:
        log.warning("  Ontologia: %s — usando fallback", exc)
        return None


def obter_constraints(ont: Optional["OntologiaPM4JUD"]) -> List[Dict]:
    """Carrega C1-C16 da ontologia. Aceita só se >= 10 regras."""
    if ont is not None:
        try:
            cs = ont.constraints_ltlf()
            if cs and len(cs) >= 10:
                log.info("  constraints_ltlf(): %d regras da ontologia", len(cs))
                return cs
            elif cs:
                log.warning("  constraints_ltlf(): apenas %d regras (< 10) — fallback", len(cs))
        except Exception as exc:
            log.warning("  constraints_ltlf() falhou: %s", exc)

    # Fallback — 16 constraints PM4JUD.owl v2.0
    log.info("  constraints_ltlf(): fallback com 16 regras embutidas")
    return [
        {"id":"C1",  "artigo":"art.91,I",      "prazo_dias":30,  "nivel":1,
         "padrao":"response_with_deadline",     "label":"HC julgado ≤30d após conclusão",
         "atividade_a":"Conclusão",             "atividade_b":"Fase externa: Julgamento"},
        {"id":"C2",  "artigo":"art.111",        "prazo_dias":2,   "nivel":2,
         "padrao":"precedence",                 "label":"MP manifesta antes da conclusão"},
        {"id":"C3",  "artigo":"art.110",        "prazo_dias":None,"nivel":2,
         "padrao":"cond_resource",              "label":"Assessor especializado na matéria"},
        {"id":"C4",  "artigo":"art.110",        "prazo_dias":None,"nivel":2,
         "padrao":"chain_response",             "label":"Preparação precede voto"},
        {"id":"C5",  "artigo":"art.110",        "prazo_dias":None,"nivel":2,
         "padrao":"precedence",                 "label":"Sem regressão após conclusão"},
        {"id":"C6",  "artigo":"PM4JUD-v2",      "prazo_dias":None,"nivel":1,
         "padrao":"response",                   "label":"HC tem resultado (443/451/12458/12475)",
         "atividade_a":"Distribuído",           "atividade_b":"Habeas Corpus Denegado"},
        {"id":"C7",  "artigo":"MetaCNJ1",       "prazo_dias":None,"nivel":1,
         "padrao":"aggregate_count",            "label":"Meta CNJ 1"},
        {"id":"C8",  "artigo":"MetaCNJ2",       "prazo_dias":None,"nivel":1,
         "padrao":"aggregate_count",            "label":"Meta CNJ 2"},
        {"id":"C9",  "artigo":"MetaCNJ4a",      "prazo_dias":None,"nivel":1,
         "padrao":"aggregate_count",            "label":"Meta CNJ 4a"},
        {"id":"C10", "artigo":"art.110,I",      "prazo_dias":10,  "nivel":1,
         "padrao":"response_with_deadline",     "label":"Decisão mono ≤10d pós conclusão",
         "atividade_a":"Conclusão para decisão","atividade_b":"Fase externa: Julgamento"},
        {"id":"C11", "artigo":"art.110,III",    "prazo_dias":30,  "nivel":1,
         "padrao":"response_with_deadline",     "label":"Voto relator ≤30d pós conclusão",
         "atividade_a":"Conclusão para decisão","atividade_b":"Publicado acórdão no DJEN"},
        {"id":"C12", "artigo":"art.95",         "prazo_dias":None,"nivel":1,
         "padrao":"response",                   "label":"Sessão após inclusão em mesa",
         "atividade_a":"Incluído em pauta",     "atividade_b":"Fase externa: Julgamento"},
        {"id":"C13", "artigo":"art.179",        "prazo_dias":30,  "nivel":1,
         "padrao":"response_with_deadline",     "label":"Acórdão ≤30d pós julgamento",
         "atividade_a":"Fase externa: Julgamento","atividade_b":"Publicado acórdão no DJEN"},
        {"id":"C14", "artigo":"art.94/95",      "prazo_dias":None,"nivel":2,
         "padrao":"precedence",                 "label":"RelVoto SAGWeb antes da sessão",
         "atividade_a":"Relatório e Voto",      "atividade_b":"Fase externa: Julgamento"},
        {"id":"C15", "artigo":"art.123/178/180","prazo_dias":None,"nivel":2,
         "padrao":"response",                   "label":"Ementa/Acórdão SAGWeb após proclamação",
         "atividade_a":"Fase externa: Julgamento","atividade_b":"Acórdão"},
        {"id":"C16", "artigo":"art.34/38",      "prazo_dias":None,"nivel":2,
         "padrao":"response",                   "label":"Despacho/Decisão SAGWeb após conclusão",
         "atividade_a":"Conclusão para decisão","atividade_b":"Despacho"},
    ]


def obter_metas_cnj(ont: Optional["OntologiaPM4JUD"]) -> List[Dict]:
    if ont is not None:
        try:
            m = ont.metas_cnj()
            if m:
                return m
        except Exception:
            pass
    return [
        {"id":"Meta1","label":"Meta CNJ 1","indicador":"J_sim/D_sim>=1.00"},
        {"id":"Meta2","label":"Meta CNJ 2","indicador":"J_acervo/N_acervo>=0.50"},
        {"id":"Meta4","label":"Meta CNJ 4a","indicador":"crimes_adm_julgados/total>=0.90"},
    ]


def resolver_atividade_uri(uri: str, mapa_tpu: Dict[int, str]) -> Optional[str]:
    """
    Converte URI de Movimento TPU (ex: mov:Movimento_51) para nome canônico.
    Extrai o código numérico da URI e busca no mapa_tpu da ontologia.
    """
    try:
        # Extrai número da URI: Movimento_51 → 51
        parte = uri.rsplit("_", 1)[-1].rstrip(">").strip()
        cod = int(parte)
        return mapa_tpu.get(cod)
    except (ValueError, IndexError):
        return None


def build_declare_model(
    constraints: List[Dict],
    mapa_tpu: Dict[int, str],
) -> Tuple[Dict[str, Any], List[Dict], List[Dict]]:
    """
    Constrói o modelo Declare no formato PM4Py a partir dos constraints da ontologia.

    Retorna
    -------
    declare_model : Dict[template, {(act_a, act_b): {}}]
        Modelo PM4Py pronto para declare_check().
    temporais : List[Dict]
        Constraints response_with_deadline — verificados separadamente com timestamps.
    ignorados : List[Dict]
        Constraints cond_resource / aggregate_count — não por-trace via Declare.
    """
    declare_model: Dict[str, Any] = {}
    temporais: List[Dict] = []
    ignorados: List[Dict] = []

    for c in constraints:
        cid    = c["id"]
        padrao = c.get("padrao", "")

        if cid in IDS_LOG_LEVEL or padrao == _PADRAO_AGREGADO:
            ignorados.append(c)
            continue

        if padrao == _PADRAO_RECURSO:
            ignorados.append(c)
            continue

        # Resolver atividades — tenta atividade_a/b embutida primeiro,
        # depois resolve via URI da ontologia usando mapa_tpu
        def _resolve_ativ(uri_key: str, sag_key: str,
                           fallback_key: str) -> Optional[str]:
            # 1. URI de MovimentoProcessual (da ontologia)
            uri_val = c.get(uri_key)
            if uri_val and ("Movimento_" in str(uri_val) or "#" in str(uri_val)):
                nome = resolver_atividade_uri(str(uri_val), mapa_tpu)
                if nome:
                    return nome
            # 2. Atividade SAGWeb (string direta da ontologia)
            sag_val = c.get(sag_key)
            if sag_val:
                return str(sag_val)
            # 3. Fallback embutido no dict do constraint
            return c.get(fallback_key)

        act_a = _resolve_ativ("atividade_a_uri", "atividade_a_sag", "atividade_a")
        act_b = _resolve_ativ("atividade_b_uri", "atividade_b_sag", "atividade_b")

        if padrao == _PADRAO_TEMPORAL:
            # Constraints bounded: adiciona ao Declare como RESPONSE (estrutural)
            # + registro separado para verificação de prazo por timestamp
            if act_a and act_b:
                t = DT.RESPONSE
                declare_model.setdefault(t, {})[(act_a, act_b)] = {}
                temporais.append({**c, "atividade_a_res": act_a,
                                       "atividade_b_res": act_b})
            continue

        template = _PADRAO_TO_TEMPLATE.get(padrao)
        if template is None:
            log.debug("  %s: padraoLTLf '%s' sem mapeamento — ignorado", cid, padrao)
            ignorados.append(c)
            continue

        if template in (DT.EXISTENCE, DT.ABSENCE, DT.INIT):
            # Templates unários: requerem exatamente 1 atividade
            if act_a:
                declare_model.setdefault(template, {})[(act_a,)] = {}
            else:
                ignorados.append(c)
        else:
            # Templates binários: requerem exatamente 2 atividades
            # PM4Py lança "tuple index out of range" se apenas 1 for fornecida
            if act_a and act_b:
                declare_model.setdefault(template, {})[(act_a, act_b)] = {}
            else:
                log.debug("  %s: template binário sem act_b — ignorado "
                          "(act_a=%s act_b=%s)", cid, act_a, act_b)
                ignorados.append(c)

    return declare_model, temporais, ignorados


# ===========================================================================
# SEÇÃO 4 — Verificadores complementares (temporais + hierarquia assuntos)
# ===========================================================================

def _ts(ev) -> Optional[datetime]:
    return ev.get("time:timestamp")

def _act(ev) -> str:
    return ev.get("concept:name", "")

def _delta_dias(t1, t2) -> float:
    if t1 is None or t2 is None:
        return float("nan")
    try:
        return (t2 - t1).total_seconds() / 86_400
    except Exception:
        return float("nan")

def _contem(nome: str, conj: Set[str]) -> bool:
    if nome in conj: return True
    nl = nome.lower()
    return any(p.lower() in nl for p in conj)

def _tem(trace: Trace, conj: Set[str]) -> bool:
    return any(_contem(_act(e), conj) for e in trace)

def _primeiro_ts(trace: Trace, conj: Set[str]) -> Optional[datetime]:
    for ev in trace:
        if _contem(_act(ev), conj):
            return _ts(ev)
    return None

def _eh_hc(trace: Trace) -> bool:
    return bool(trace.attributes.get(ATTR_PRIO, False))


def verificar_deadline(trace: Trace, c: Dict) -> bool:
    """
    Verifica a bounded response: se atividade_a ocorre, atividade_b deve
    ocorrer dentro de prazo_dias. Retorna True se conforme ou não aplicável.
    """
    prazo = c.get("prazo_dias")
    if not prazo or prazo <= 0:
        return True   # sem prazo definido → só a estrutura (já verificada pelo Declare)

    act_a_set = {c.get("atividade_a_res", "")}
    act_b_set = {c.get("atividade_b_res", "")}

    evs = list(trace)
    for i, ev in enumerate(evs[:-1]):
        if not _contem(_act(ev), act_a_set):
            continue
        ts_a = _ts(ev)
        if ts_a is None:
            continue
        # Próxima ocorrência de B
        for ev2 in evs[i+1:]:
            if _contem(_act(ev2), act_b_set):
                d = _delta_dias(ts_a, _ts(ev2))
                if not (d != d) and d > prazo:
                    return False
                break
    return True


def inicializar_hierarquia(ont: Optional["OntologiaPM4JUD"]) -> None:
    """
    Pré-computa conjuntos de códigos TPU por categoria (rdfs:subClassOf*).
    Usado para a Meta CNJ 4a (crimes contra a administração pública).
    """
    global _CATS
    if ont is None:
        _CATS = {}
        return
    try:
        _CATS = ont.categorias_assunto()
        total = sum(len(v) for v in _CATS.values())
        if total:
            log.info("  Hierarquia assuntos: %d categorias | %d códigos TPU",
                     len(_CATS), total)
        else:
            log.warning("  Hierarquia assuntos: PM4JUD_Assuntos.owl não carregado "
                        "— Meta CNJ 4a usará fallback (HC/RHC)")
    except Exception as exc:
        log.warning("  Hierarquia assuntos: %s", exc)
        _CATS = {}


def _eh_crime_adm_publica(trace: Trace) -> Optional[bool]:
    """
    Verifica se o traço tem algum assunto classificado como
    'crime contra a administração pública' (Título XI CP).

    O atributo pm4jud:assuntos contém todos os assuntos do processo
    no formato "código:Nome; código2:Nome2; ..." (separados por ';').
    Retorna True se QUALQUER código do traço estiver nos 66 códigos
    carregados via SPARQL (ehCrimeAdministracaoPublica=true).
    Retorna None se a hierarquia não estiver carregada ou o atributo
    não existir no traço.
    """
    codigos = _CATS.get("crimes_adm_publica")
    if not codigos:
        return None
    assuntos_str = trace.attributes.get(ATTR_ASSUNTOS)
    if not assuntos_str:
        return None
    # Parsear "4355:Prisão Preventiva; 3568:Corrupção ativa; ..."
    for item in str(assuntos_str).split(";"):
        parte_codigo = item.strip().split(":")[0].strip()
        try:
            if int(parte_codigo) in codigos:
                return True
        except (ValueError, TypeError):
            pass
    return False


# ===========================================================================
# SEÇÃO 5 — Metas CNJ (η) — nível de log
# ===========================================================================

def calcular_eta(event_log: EventLog,
                 current_state: Optional[Dict],
                 params_des: Optional[List[Dict]]) -> Dict:
    """Calcula η = aderência Metas CNJ 1, 2 e 4a."""
    ref_ano = datetime.now(timezone.utc).year - 1   # 2024

    n_ingressados = 0; n_julgados_ano = 0
    n_antigos = 0; n_ant_julg = 0
    n_crim_adm = 0; n_crim_adm_julg = 0
    hierarquia_ok = bool(_CATS.get("crimes_adm_publica"))
    corte = datetime(ref_ano, 1, 1, tzinfo=timezone.utc)

    for trace in event_log:
        ts_dist = _primeiro_ts(trace, _DIST)
        ts_julg = _primeiro_ts(trace, _SESSAO | _ACORDAO | _TRAN | _BAIXA)

        # Meta 1
        if ts_dist and ts_dist.year == ref_ano:
            n_ingressados += 1
            if ts_julg: n_julgados_ano += 1

        # Meta 2 — antigos
        if ts_dist and ts_dist <= corte:
            n_antigos += 1
            if ts_julg: n_ant_julg += 1

        # Meta 4a — crimes adm. pública (com hierarquia) ou HC/RHC (fallback)
        if hierarquia_ok:
            pert = _eh_crime_adm_publica(trace)
            if pert is True:
                n_crim_adm += 1
                if ts_julg: n_crim_adm_julg += 1
        else:
            if _eh_hc(trace):
                n_crim_adm += 1
                if ts_julg: n_crim_adm_julg += 1

    # Complementar n_antigos com current_state do P5
    if current_state:
        for caso in current_state.get("casos", []):
            try:
                ts = datetime.fromisoformat(caso.get("timestamp",""))
                if ts.tzinfo is None: ts = ts.replace(tzinfo=timezone.utc)
                if ts <= corte: n_antigos += 1
            except Exception: pass

    # Calcular por fallback de λ se n_ingressados=0
    if n_ingressados == 0 and params_des:
        try:
            lam = float(params_des[0].get("lambda_proc_mes", 0))
            n_ingressados = round(lam * 12)
            n_julgados_ano = sum(
                1 for t in event_log
                if _primeiro_ts(t, _SESSAO | _ACORDAO | _TRAN) is not None
            )
        except Exception: pass

    meta1_val = round(n_julgados_ano / n_ingressados, 4) if n_ingressados else 0.0
    meta2_val = round(n_ant_julg    / n_antigos,     4) if n_antigos      else 1.0
    meta4_val = round(n_crim_adm_julg / n_crim_adm,  4) if n_crim_adm     else 1.0

    eta = round((int(meta1_val>=1.0) + int(meta2_val>=0.5) + int(meta4_val>=0.9)) / 3, 4)

    return {
        "eta":   eta,
        "meta1": {"valor": meta1_val, "conforme": meta1_val>=1.0,
                  "n_julgados": n_julgados_ano, "n_ingressados": n_ingressados,
                  "threshold": 1.00, "artigo": "MetaCNJ1"},
        "meta2": {"valor": meta2_val, "conforme": meta2_val>=0.5,
                  "n_antigos_julgados": n_ant_julg, "n_antigos": n_antigos,
                  "threshold": 0.50, "artigo": "MetaCNJ2"},
        "meta4": {"valor": meta4_val, "conforme": meta4_val>=0.9,
                  "n_crimes_adm_julgados": n_crim_adm_julg,
                  "n_crimes_adm_total": n_crim_adm,
                  "hierarquia_assuntos": hierarquia_ok,
                  "threshold": 0.90, "artigo": "MetaCNJ4a"},
    }


# ===========================================================================
# SEÇÃO 6 — Pipeline por gabinete
# ===========================================================================

def carregar_json(fpath: Path, gabinete: str, desc: str) -> Optional[Dict]:
    if not fpath.exists():
        log.warning("  %s não encontrado", fpath.name)
        return None
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            data = json.loads(fpath.read_text(encoding=enc))
            log.info("  %s: carregado [%s]", desc, enc)
            return data
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    log.warning("  %s: não foi possível ler", fpath.name)
    return None


def carregar_params_des(input_dir: Path, gabinete: str) -> Optional[List[Dict]]:
    fpath = input_dir / f"params_des_{gabinete}.csv"
    if not fpath.exists():
        return None
    try:
        with fpath.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        log.info("  params_des: %d atividades (P5)", len(rows))
        return rows
    except Exception as exc:
        log.warning("  params_des: %s", exc)
        return None


def processar_gabinete(gabinete: str,
                       input_dir: Path,
                       output_dir: Path,
                       constraints: List[Dict],
                       metas_meta: List[Dict],
                       ont: Optional["OntologiaPM4JUD"]) -> Dict:

    log.info("=" * 70)
    log.info("Processando gabinete: %s", gabinete.upper())
    log.info("=" * 70)
    t0  = time.time()
    res = {"gabinete": gabinete, "status": "OK"}

    # --- XES (mesmo input do P5 — vem do P4) ---
    xes_path = input_dir / f"refine2_{gabinete}.xes"
    if not xes_path.exists():
        log.error("  Arquivo não encontrado: %s", xes_path)
        return {**res, "status": "ERRO", "motivo": "XES ausente"}

    log.info("  Carregando log: %s", xes_path.name)
    event_log = xes_importer.apply(str(xes_path))
    n_traces  = len(event_log)
    n_eventos = sum(len(t) for t in event_log)
    n_tpu     = sum(1 for t in event_log for e in t if not e.get(ATTR_SIM))
    n_sag     = n_eventos - n_tpu
    fase      = "Fase-1 (sintético)" if n_sag > 0 else "Fase-2 (real)"
    log.info("  Log: %d traços | %d eventos | %d TPU | %d SAGWeb | %s",
             n_traces, n_eventos, n_tpu, n_sag, fase)

    # --- Hierarquia assuntos (rdfs:subClassOf*) ---
    inicializar_hierarquia(ont)

    # --- Artefatos P5 ---
    current_state = carregar_json(
        input_dir / f"current_state_{gabinete}.json", gabinete, "current_state")
    params_des = carregar_params_des(input_dir, gabinete)

    # --- Mapa TPU para resolver URIs dos constraints ---
    mapa_tpu: Dict[int, str] = {}
    if ont is not None:
        try:
            mapa_tpu = ont.mapa_tpu()
            log.info("  mapa_tpu: %d movimentos carregados", len(mapa_tpu))
        except Exception as exc:
            log.warning("  mapa_tpu: %s", exc)

    # --- Construir modelo Declare ---
    declare_model, temporais, ignorados = build_declare_model(constraints, mapa_tpu)

    n_declare   = sum(len(v) for v in declare_model.values())
    n_temporais = len(temporais)
    n_ignorados = len(ignorados)
    log.info("  Modelo Declare: %d constraints | %d temporais | %d ignorados",
             n_declare, n_temporais, n_ignorados)
    for tmpl, rules in declare_model.items():
        for atos in rules:
            log.debug("    %s: %s", tmpl, " → ".join(atos))

    # --- Conformance checking PM4Py Declare ---
    log.info("  Executando declare_check() [PM4Py]...")
    t1 = time.time()
    try:
        cc_results = declare_check(event_log, declare_model)
    except Exception as exc:
        log.error("  declare_check() falhou: %s", exc)
        cc_results = [{"dev_fitness": 1.0, "no_dev_total": 0,
                       "no_constr_total": n_declare, "deviations": []}
                      for _ in range(n_traces)]
    log.info("  declare_check(): %.1fs", time.time() - t1)

    # --- Verificação de prazo (temporais) ---
    n_prazo_viols = 0
    prazo_viols_por_c: Dict[str, int] = {c["id"]: 0 for c in temporais}
    for trace in event_log:
        for c in temporais:
            if not verificar_deadline(trace, c):
                prazo_viols_por_c[c["id"]] = prazo_viols_por_c.get(c["id"], 0) + 1
                n_prazo_viols += 1

    # --- κ: média ponderada de dev_fitness (Declare) + conformidade temporal ---
    fitnesses = [r.get("dev_fitness", 1.0) for r in cc_results]

    # Ajustar fitness por violações de prazo (penaliza proporcionalmente)
    if temporais:
        total_checks_temp = n_traces * len(temporais)
        taxa_prazo = 1.0 - (n_prazo_viols / total_checks_temp) if total_checks_temp else 1.0
        # Peso: Declare = n_declare/(n_declare+len(temporais)); temporal = len(temporais)/(...)
        w_dec  = n_declare   / (n_declare + len(temporais)) if (n_declare + len(temporais)) else 1.0
        w_temp = len(temporais) / (n_declare + len(temporais)) if (n_declare + len(temporais)) else 0.0
        kappa  = round(float(np.mean(fitnesses)) * w_dec + taxa_prazo * w_temp, 4)
    else:
        kappa = round(float(np.mean(fitnesses)), 4)

    n_viols = sum(1 for r in cc_results if r.get("no_dev_total", 0) > 0)
    log.info("  κ = %.4f  (%d/%d traços com desvios Declare | %d viols prazo)",
             kappa, n_viols, n_traces, n_prazo_viols)

    # Desvios por tipo de constraint
    desvios_por_template: Dict[str, int] = {}
    for r in cc_results:
        for dev in r.get("deviations", []):
            chave = str(dev[0]) if dev else "?"
            desvios_por_template[chave] = desvios_por_template.get(chave, 0) + 1

    for tmpl, cnt in sorted(desvios_por_template.items()):
        # cnt = ocorrências totais por template (pode ser > n_traces se múltiplas
        # violações por trace). Taxa = fração de traces com pelo menos 1 desvio.
        traces_com_desvio_tmpl = sum(
            1 for r in cc_results
            if any(str(dev[0]) == tmpl for dev in r.get("deviations", []))
        )
        taxa = 1.0 - traces_com_desvio_tmpl / n_traces if n_traces else 1.0
        est  = "✓" if taxa >= 1.0 else ("△" if taxa >= 0.8 else "✗")
        log.info("    %s %-20s %.1f%%  (%d/%d traços)",
                 est, tmpl, taxa*100, traces_com_desvio_tmpl, n_traces)
    for cid, cnt in sorted(prazo_viols_por_c.items()):
        taxa = 1.0 - cnt / n_traces
        est  = "✓" if taxa >= 1.0 else ("△" if taxa >= 0.8 else "✗")
        c_meta = next((c for c in temporais if c["id"]==cid), {})
        log.info("    %s %-4s (prazo ≤%dd)  %.1f%%",
                 est, cid, c_meta.get("prazo_dias", 0), taxa*100)

    # --- η: Metas CNJ ---
    log.info("  Calculando Metas CNJ (η)...")
    eta_res = calcular_eta(event_log, current_state, params_des)
    eta = eta_res["eta"]
    log.info("  η = %.4f  (Meta1=%s Meta2=%s Meta4a=%s)",
             eta,
             "✓" if eta_res["meta1"]["conforme"] else "✗",
             "✓" if eta_res["meta2"]["conforme"] else "✗",
             "✓" if eta_res["meta4"]["conforme"] else "✗")

    # --- Salvar ltlf_<gab>.json ---
    payload = {
        "programa":       "PM4JUD-LTLf",
        "versao":         VERSION,
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "gabinete":       gabinete,
        "fase":           fase,
        "n_traces":       n_traces,
        "n_eventos":      n_eventos,
        "n_tpu":          n_tpu,
        "n_sagweb":       n_sag,
        # κ
        "kappa":          kappa,
        "kappa_label":    "Taxa de conformidade regimental (κ) — PM4Py Declare",
        "kappa_declare":  round(float(np.mean(fitnesses)), 4),
        "kappa_temporal": round(1.0 - n_prazo_viols/(n_traces*len(temporais)), 4) if temporais and n_traces else 1.0,
        "n_traces_com_desvio":   n_viols,
        "desvios_por_template":  desvios_por_template,
        "violacoes_prazo_por_c": prazo_viols_por_c,
        # η
        "eta":            eta,
        "eta_label":      "Aderência Metas Nacionais CNJ (η)",
        "metas_cnj":      eta_res,
        # Modelo Declare usado
        "modelo_declare": {
            tmpl: [list(atos) for atos in rules]
            for tmpl, rules in declare_model.items()
        },
        "constraints_metadata": [
            {k: v for k, v in c.items() if k != "uri"}
            for c in constraints
        ],
    }
    fpath = output_dir / f"ltlf_{gabinete}.json"
    fpath.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                     encoding="utf-8")
    log.info("  Salvo: %s", fpath.name)

    res.update({
        "n_traces":    n_traces,
        "fase":        fase,
        "kappa":       kappa,
        "eta":         eta,
        "n_viols":     n_viols,
        "tempo_s":     round(time.time() - t0, 1),
    })
    log.info("  %s: κ=%.4f | η=%.4f | desvios=%d/%d | t=%.0fs",
             gabinete, kappa, eta, n_viols, n_traces, res["tempo_s"])
    return res


# ===========================================================================
# SEÇÃO 7 — Relatório consolidado e entry point
# ===========================================================================

def salvar_relatorio(resultados: List[Dict],
                     output_dir: Path,
                     constraints_meta: List[Dict]) -> None:
    kappas = [r["kappa"] for r in resultados if "kappa" in r]
    etas   = [r["eta"]   for r in resultados if "eta"   in r]
    payload = {
        "programa":   "PM4JUD-LTLf",
        "versao":     VERSION,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "proximo_passo": "PM4JUD-DES (P7) — usa κ como restrição hard/soft",
        "gabinetes":  resultados,
        "sumario": {
            "kappa_medio": round(float(np.mean(kappas)), 4) if kappas else None,
            "kappa_min":   round(float(np.min(kappas)),  4) if kappas else None,
            "eta_medio":   round(float(np.mean(etas)),   4) if etas   else None,
            "n_gabinetes": len(resultados),
            "motor":       "PM4Py Declare nativo + verificador de prazo",
        },
        "constraints_metadata": [
            {k: v for k, v in c.items() if k != "uri"}
            for c in constraints_meta
        ],
    }
    fpath = output_dir / "p6_relatorio.json"
    fpath.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                     encoding="utf-8")
    log.info("Relatório: %s", fpath.name)


def imprimir_resumo(resultados: List[Dict]) -> None:
    log.info("")
    log.info("=" * 65)
    log.info("RESUMO P6 — CONFORMIDADE REGIMENTAL (PM4Py Declare)")
    log.info("=" * 65)
    log.info("  %-12s  %8s  %8s  %9s", "Gabinete", "κ (RISTJ)", "η (CNJ)", "Desvios")
    log.info("  %s", "-"*50)
    for r in resultados:
        log.info("  %-12s  %8.4f  %8.4f  %4d/%-4d",
                 r["gabinete"], r.get("kappa",0), r.get("eta",0),
                 r.get("n_viols",0), r.get("n_traces",0))
    kappas = [r["kappa"] for r in resultados if "kappa" in r]
    if kappas:
        log.info("")
        log.info("  κ médio = %.4f | η médio = %.4f",
                 np.mean(kappas),
                 np.mean([r.get("eta",0) for r in resultados]))
    log.info("")
    log.info("Próximo: PM4JUD-DES (P7)")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pm4jud_ltlf",
        description="PM4JUD-LTLf v2.0 (P6) — PM4Py Declare nativo",
    )
    parser.add_argument("--input",     required=True, type=Path)
    parser.add_argument("--output",    required=True, type=Path)
    parser.add_argument("--ontologia", default=None,  type=Path)
    parser.add_argument("--gabinetes", nargs="+",
                        default=["reynaldo", "palheiro", "schietti"])
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    log.info("PM4JUD-LTLf v%s | motor: PM4Py Declare nativo", VERSION)
    log.info("Gabinetes: %s", args.gabinetes)

    ont         = carregar_ontologia(args.ontologia)
    constraints = obter_constraints(ont)
    metas_meta  = obter_metas_cnj(ont)
    log.info("Constraints: %d (C1-C16) | %d log-level (C7/C8/C9)",
             len(constraints), len(IDS_LOG_LEVEL))

    resultados = []
    for gabinete in args.gabinetes:
        r = processar_gabinete(gabinete, args.input, args.output,
                               constraints, metas_meta, ont)
        resultados.append(r)

    salvar_relatorio(resultados, args.output, constraints)
    imprimir_resumo(resultados)


if __name__ == "__main__":
    main()
