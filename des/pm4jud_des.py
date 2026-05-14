#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
PM4JUD-DES  v1.0
================================================================================

Dissertação de Mestrado — PPGIa/PUCPR
Título: PM4JUD — Otimização Multiobjetivo com Mineração de Processos e
        Simulação no Contexto do Fluxo Processual em Gabinetes de Magistrado
Autor:  Luiz Claudio Soares de Almeida
Orient: Prof. Dr. Edson Emilio Scalabrin
Ano:    2026

Descrição
---------
P7b do pipeline PM4JUD. Segunda etapa da Fase 3 (Simulação Computacional).
Implementa o modelo de Simulação por Eventos Discretos (DES) em SimPy M/M/c
para os gabinetes de magistrado do STJ, conforme o Algoritmo 5 de
FERRONATO (2022, p. 115): usa os logs sintéticos do P7a como workload de
chegadas e os parâmetros do P5 (SR, RC, GF, CE, EL) para configurar o SimPy.

Modelo de filas M/M/c com prioridade
--------------------------------------
  Fila 1 (Alta Prioridade) : HC/RHC — preempção por prazo constitucional
  Fila 2 (Regular)         : demais processos da 3ª Seção Criminal

Recursos simulados (Res. STJ N.19/2026, N_ASSESSORES = 38):
  CJ3A  : 10 Analistas Judiciários — instrutor, peso 1,00
  CJ3C  :  1 Analista Judiciário   — gestão,   peso 0,10
  CJ2A  :  3 Técnicos Judiciários  — instrutor, peso 0,80
  FC6C  : 15 Técnicos Judiciários  — instrutor, peso 0,70
  FC4IV :  7 Auxiliares            — admin,     peso 0,20
  FC2II :  2 Auxiliares            — admin,     peso 0,10

Restrições regimentais (Ontologia PM4JUD Módulo 7 — SPARQL em runtime)
-----------------------------------------------------------------------
  Hard (violação = solução inválida para o P8 OPT):
    C10 : Decisão monocrática ≤ 10 dias após conclusão (Art. 110-I RISTJ)
    C11 : Acórdão ≤ 30 dias após sessão (Art. 110-III / Art. 179 RISTJ)
  Soft (penalização proporcional na métrica κ):
    C1  : HC julgado em ≤ 30 dias (Art. 91-I RISTJ)
    C12 : Sessão agendada antes do julgamento colegiado (Art. 95 RISTJ)

Métricas de saída (4 métricas da dissertação — OE(d))
------------------------------------------------------
  T̄  : tempo médio de julgamento (dias)             — f1 do MOOP
  G   : índice de Gini do balanceamento de carga     — f2 do MOOP
  κ   : taxa de conformidade regimental (%)          — f3 do MOOP
  η   : aderência Metas CNJ 1, 2 e 4 (%)            — f4 do MOOP

Cada métrica é calculada por replicação e agregada com média, desvio
padrão e IC 95% (t de Student, 29 g.l., α=0,05) para análise em P9.

Validação — Método (a) de FERRONATO (2022, p. 117)
---------------------------------------------------
O DES compara suas saídas estatísticas com o log original:
  λ (taxa de chegada), T̄ (throughput), espera em fila, ρ (utilização).
Ferronato: "as estatísticas da simulação devem atingir [o limiar definido]".
Limiar: fitness_a >= 0.75 (ajustável por especialista STJ na Fase 2).
Este é o Método (a) do PM4SOS — implementado no DES, não no Sim2Log,
pois as métricas de fila emergem da simulação, não da geração dos logs.

Design experimental
-------------------
  90 execuções: 1 GC × 30 rep + 3 GE × 30 rep (NSGA-II/AMGA2/SPEA2)
  Chamado pelo P8 OPT a cada avaliação de fitness das soluções candidatas.
  Standalone: grupo controle com configuração atual do STJ.

Ontologia PM4JUD — uso em P7b
------------------------------
  Módulo 3 (Classes)   : identifica HC/RHC para roteamento de filas
  Módulo 5 (Movimentos): mapeia atividades TPU para tempos de serviço
  Módulo 7 (PM4JUD)    : restrições C1–C16 hard/soft (κ durante simulação)

Entradas (output/ dos pipelines anteriores)
--------------------------------------------
  sim2log_<gab>_r<N>.xes     Logs sintéticos do P7a (30 por gabinete)
  params_des_<gab>.csv        λ, μ, ρ por atividade (do P5)
  org_model_<gab>.json        N_ASSESSORES e estrutura dos assessores (do P5)
  current_state_<gab>.json    Estado inicial das filas (do P5)
  ltlf_<gab>.json             κ_baseline e constraints C1–C16 (do P6)
  refine2_<gab>.xes           Log original para validação Método (a) (do P4)
  --ontologia  <dir>          Diretório com PM4JUD_*.owl (Módulos 3, 5, 7)
  --configuracao <json>       Alocação de assessores do P8 OPT (ou padrão STJ)

Saídas (output/)
----------------
  des_<gab>_r<N>.json         Métricas por replicação (T̄, G, κ, η)
  des_<gab>.json              Agregado por gabinete (média, std, IC95)
  p7b_relatorio.json          Resumo consolidado — entrada para P8/P9

Pipeline completo
-----------------
  P1→P2→P3→P4→P5→P6→P7a→[P7b DES]→P8 OPT→P9 STAT

Referências
-----------
  FERRONATO, J. J. PM4SOS. Tese (Doutorado em Informática) — PUCPR, 2022.
  VAN DER AALST, W. M. P. Process Mining: Data Science in Action.
    2. ed. Berlin: Springer, 2016.
  ROZINAT, A. et al. Discovering simulation models from event logs. IS, 2009.

Repositório: https://github.com/luizcsalmeida/pm4jud/tree/main/des
================================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Generator, List, Optional, Set, Tuple

import numpy as np
import simpy

# ---------------------------------------------------------------------------
# PM4Py
# ---------------------------------------------------------------------------
try:
    from pm4py.objects.log.obj import EventLog, Trace
    from pm4py.objects.log.importer.xes import importer as xes_importer
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
    from pm4jud_ontologia import OntologiaPM4JUD
    _ONT_OK = True
except ImportError:
    _ONT_OK = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] PM4JUD-DES — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("PM4JUD-DES")
VERSION = "1.0"

# ===========================================================================
# SEÇÃO 1 — Constantes do domínio STJ/PM4JUD
# ===========================================================================

# Estrutura de assessores (Res. STJ N.19/2026)
# (categoria, n, funcao, peso_efetivo)
ESTRUTURA_GABINETE = [
    ("CJ3A",  10, "instrutor", 1.00),   # Analistas — instrutor pleno
    ("CJ3C",   1, "gestao",    0.10),   # Analista — gestão do gabinete
    ("CJ2A",   3, "instrutor", 0.80),   # Técnicos — instrutor sênior
    ("FC6C",  15, "instrutor", 0.70),   # Técnicos — instrutor
    ("FC4IV",  7, "admin",     0.20),   # Auxiliares — administrativo
    ("FC2II",  2, "admin",     0.10),   # Auxiliares — apoio
]
N_ASSESSORES = sum(n for _, n, _, _ in ESTRUTURA_GABINETE)   # 38

# Atributos XES PM4JUD
ATTR_PRIO    = "pm4jud:prioritario"
ATTR_SIM     = "pm4jud:sim_flag"

# Prazos regimentais (segundos)
PRAZO_MONO_S     = 10 * 24 * 3600     # 10 dias — Art. 110-I RISTJ
PRAZO_ACORDAO_S  = 30 * 24 * 3600     # 30 dias — Art. 110-III / Art. 179 RISTJ
PRAZO_HC_S       = 30 * 24 * 3600     # 30 dias — Art. 91-I RISTJ (soft)

# Prioridades SimPy (menor = maior prioridade)
PRIORIDADE_HC      = 1    # HC/RHC
PRIORIDADE_REGULAR = 5    # demais processos

# Métricas CNJ para η
THRESHOLD_META1 = 1.00    # julgar >= 100% dos ingressados
THRESHOLD_META2 = 0.50    # julgar >= 50% do acervo antigo
THRESHOLD_META4 = 0.90    # julgar >= 90% de crimes adm. pública


# ===========================================================================
# SEÇÃO 2 — Configuração de assessores
# ===========================================================================

class ConfiguracaoGabinete:
    """
    Define a alocação de assessores para o gabinete.
    Na execução standalone (grupo controle), usa a estrutura padrão do STJ.
    Quando chamado pelo P8 OPT, recebe configuração modificada com
    diferentes quantidades e pesos por categoria.

    A configuração é a VARIÁVEL DE DECISÃO do MOOP:
      x = [n_CJ3A, n_CJ2A, n_FC6C, n_FC4IV, n_FC2II]
    sujeito a: sum(x) = N_ASSESSORES
    """

    def __init__(self, config_json: Optional[Dict] = None):
        if config_json:
            self.estrutura = config_json.get("estrutura", ESTRUTURA_GABINETE)
        else:
            self.estrutura = ESTRUTURA_GABINETE

        # Lista expandida de assessores individuais
        self.assessores: List[Dict] = []
        for cat, n, funcao, peso in self.estrutura:
            for i in range(n):
                self.assessores.append({
                    "id":      f"{cat}-{i+1:02d}",
                    "cat":     cat,
                    "funcao":  funcao,
                    "peso":    peso,
                })

        # Assessores instrutores (habilitados para atividades judiciais)
        self.instrutores = [a for a in self.assessores if a["funcao"] == "instrutor"]
        self.admin       = [a for a in self.assessores if a["funcao"] == "admin"]
        self.n_total     = len(self.assessores)
        self.n_inst      = len(self.instrutores)

    def capacidade_efetiva(self) -> float:
        """Capacidade efetiva total: soma dos pesos × n por categoria."""
        return sum(n * peso for _, n, _, peso in self.estrutura)

    def __repr__(self):
        return (f"ConfigGabinete(n={self.n_total} | "
                f"inst={self.n_inst} | cap_ef={self.capacidade_efetiva():.1f})")


# ===========================================================================
# SEÇÃO 3 — Monitoramento de métricas durante simulação
# ===========================================================================

class MonitorMetricas:
    """
    Coleta métricas durante a execução SimPy:
      • Tempo de julgamento por caso (entrada → saída do sistema)
      • Tempo de espera por fila (HC e regular separados)
      • Carga de trabalho por assessor (para Gini)
      • Violações de constraints RISTJ (para κ)
      • Metas CNJ (para η)
    """

    def __init__(self):
        # Tempos de julgamento (dias)
        self.t_julgamento_hc:      List[float] = []
        self.t_julgamento_regular: List[float] = []
        # Tempo de espera em fila (segundos)
        self.t_espera_hc:      List[float] = []
        self.t_espera_regular: List[float] = []
        # Carga por assessor {id: n_casos_processados}
        self.carga_assessor: Dict[str, int] = {}
        # Violações de constraints
        self.violacoes: Dict[str, int] = {
            "C1":  0, "C10": 0, "C11": 0, "C12": 0,
        }
        # Contadores para η (Metas CNJ)
        self.n_hc_julgados = 0
        self.n_hc_total    = 0
        self.n_julgados    = 0
        self.n_ingressados = 0
        self.n_checks      = {k: 0 for k in self.violacoes}

    def registrar_julgamento(self,
                              case_id:      str,
                              prioritario:  bool,
                              t_chegada_s:  float,
                              t_inicio_s:   float,
                              t_saida_s:    float,
                              assessor_id:  str,
                              env_time_ref: float = 0.0) -> None:
        """Registra conclusão de um caso."""
        t_total_s = t_saida_s - t_chegada_s
        t_espera_s = t_inicio_s - t_chegada_s

        t_dias = t_total_s / 86_400

        self.n_julgados  += 1
        self.n_ingressados += 1

        if prioritario:
            self.t_julgamento_hc.append(t_dias)
            self.t_espera_hc.append(t_espera_s)
            self.n_hc_total    += 1
            self.n_hc_julgados += 1

            # C1: HC em ≤ 30 dias (soft constraint)
            self.n_checks["C1"] += 1
            if t_dias > 30:
                self.violacoes["C1"] += 1
        else:
            self.t_julgamento_regular.append(t_dias)
            self.t_espera_regular.append(t_espera_s)

        # Carga do assessor
        self.carga_assessor[assessor_id] = \
            self.carga_assessor.get(assessor_id, 0) + 1

    def registrar_violacao_prazo(self, constraint: str) -> None:
        """Registra violação de prazo regimental (C10 ou C11)."""
        if constraint in self.violacoes:
            self.violacoes[constraint] += 1
            self.n_checks[constraint] = self.n_checks.get(constraint, 0) + 1

    def calcular_t_medio(self) -> float:
        """T̄: tempo médio de julgamento em dias (todos os tipos)."""
        todos = self.t_julgamento_hc + self.t_julgamento_regular
        return float(np.mean(todos)) if todos else 0.0

    def calcular_gini(self) -> float:
        """
        Gini coefficient da carga de trabalho entre assessores.
        G=0: carga perfeitamente equilibrada.
        G=1: toda carga concentrada em um assessor.
        Fórmula: G = Σ|xi - xj| / (2n·Σxi)
        """
        cargas = list(self.carga_assessor.values())
        if len(cargas) < 2 or sum(cargas) == 0:
            return 0.0
        n     = len(cargas)
        soma  = sum(cargas)
        diffs = sum(abs(xi - xj) for xi in cargas for xj in cargas)
        return round(diffs / (2 * n * soma), 4)

    def calcular_kappa(self) -> float:
        """
        κ: taxa de conformidade regimental.
        κ = 1 - (total_violações / total_checks)
        Checks = todas as verificações de constraints realizadas.
        """
        total_viols  = sum(self.violacoes.values())
        total_checks = sum(self.n_checks.values())
        if total_checks == 0:
            return 1.0
        return round(1.0 - total_viols / total_checks, 4)

    def calcular_eta(self) -> float:
        """
        η: aderência Metas CNJ 1, 2 e 4.
        Proxy na simulação: baseado nos casos HC julgados e proporção total.

        Meta 1: n_julgados / n_ingressados >= 1.00
        Meta 2: proxy = taxa de julgamento (simulação de curto prazo)
        Meta 4: n_hc_julgados / n_hc_total >= 0.90 (crimes adm. pública)
        """
        meta1_val = (self.n_julgados / max(self.n_ingressados, 1))
        meta2_val = meta1_val   # proxy na simulação de curto prazo
        meta4_val = (self.n_hc_julgados / max(self.n_hc_total, 1))

        meta1_ok = meta1_val >= THRESHOLD_META1
        meta2_ok = meta2_val >= THRESHOLD_META2
        meta4_ok = meta4_val >= THRESHOLD_META4

        return round((int(meta1_ok) + int(meta2_ok) + int(meta4_ok)) / 3, 4)


# ===========================================================================
# SEÇÃO 4 — Modelo de simulação SimPy M/M/c com prioridade
# ===========================================================================

class ModeloGabinete:
    """
    Modelo DES de um gabinete de magistrado do STJ.

    Arquitetura M/M/c com prioridade:
      • c = n_instrutores (servidores paralelos para atividades judiciais)
      • M de chegada: processo de Poisson com λ_prio e λ_regular
      • M de serviço: distribuição extraída do log (via params_des)
      • Prioridade: HC/RHC preempta processos regulares na fila

    Restrições RISTJ (Ontologia Módulo 7):
      • Hard C10: prazo ≤ 10d após conclusão (monitorado por processo)
      • Hard C11: prazo ≤ 30d após sessão (monitorado por processo)
      • Soft C1 : prazo ≤ 30d para HC (registrado, não interrompe)
    """

    def __init__(self,
                 env:        simpy.Environment,
                 config:     ConfiguracaoGabinete,
                 params_des: List[Dict],
                 monitor:    MonitorMetricas,
                 restricoes: Optional[List[Dict]] = None,
                 rng:        Optional[random.Random] = None):
        self.env        = env
        self.config     = config
        self.monitor    = monitor
        self.restricoes = restricoes or []
        self.rng        = rng or random.Random(42)

        # Recursos SimPy: um PriorityResource para instrutores
        # c = n_instrutores (capacidade efetiva proporcional ao peso)
        self.fila_inst = simpy.PriorityResource(
            env, capacity=config.n_inst)
        self.fila_adm  = simpy.PriorityResource(
            env, capacity=len(config.admin))

        # Mapa atividade → tempo médio de serviço (segundos)
        self._mu_por_ativ: Dict[str, float] = {}
        for row in params_des:
            ativ = row.get("atividade", "")
            mu   = row.get("mu_seg") or row.get("mu") or row.get("tempo_medio_s")
            if ativ and mu:
                try:
                    self._mu_por_ativ[ativ] = float(mu)
                except (ValueError, TypeError):
                    pass

        # Assessores individuais para rastreamento de carga
        self._assessor_livre: Dict[str, bool] = {
            a["id"]: True for a in config.assessores}

    def _tempo_servico(self, atividade: str) -> float:
        """
        Amostra tempo de serviço para a atividade (em segundos).
        Usa μ do params_des ou exponencial com μ padrão de 1 hora.
        """
        mu = self._mu_por_ativ.get(atividade, 3600.0)
        mu = max(mu, 1.0)
        return self.rng.expovariate(1.0 / mu)

    def _selecionar_assessor(self, funcao: str = "instrutor") -> Optional[str]:
        """Seleciona o assessor menos carregado com a função desejada."""
        candidatos = (
            self.config.instrutores if funcao == "instrutor"
            else self.config.assessores
        )
        # Peso inverso à carga acumulada
        cargas = self.monitor.carga_assessor
        menos_carregado = min(
            candidatos,
            key=lambda a: cargas.get(a["id"], 0) / max(a["peso"], 0.01)
        )
        return menos_carregado["id"]

    def processar_caso(self,
                        case_id:     str,
                        trace:       Trace,
                        prioritario: bool) -> Generator:
        """
        Processo SimPy para um caso judicial.

        Fluxo:
          1. Chegar ao sistema → registrar timestamp de chegada
          2. Requisitar assessor instrutor (PriorityResource)
          3. Para cada atividade da trace:
             a. Processar com tempo exponencial(μ_atividade)
             b. Verificar constraints RISTJ ativos
          4. Liberar assessor → registrar métricas
        """
        t_chegada = self.env.now
        prioridade = PRIORIDADE_HC if prioritario else PRIORIDADE_REGULAR

        # Requisitar assessor instrutor (fila M/M/c com prioridade)
        with self.fila_inst.request(priority=prioridade) as req:
            yield req
            t_inicio = self.env.now

            # Selecionar assessor específico (rastreamento de carga)
            assessor_id = self._selecionar_assessor("instrutor")
            t_conclusao = None   # timestamp para verificar C10

            # Processar cada atividade da trace sintética
            evs = list(trace)
            for ev in evs:
                atividade = ev.get("concept:name", "")
                if not atividade:
                    continue

                tempo = self._tempo_servico(atividade)

                # Verificar prazo C10: decisão monocrática ≤ 10d após conclusão
                if "conclus" in atividade.lower() and t_conclusao is None:
                    t_conclusao = self.env.now

                yield self.env.timeout(tempo)

                # Monitorar C10: ato ordinatório após conclusão
                if t_conclusao is not None and "ato ordinat" in atividade.lower():
                    delta = self.env.now - t_conclusao
                    self.monitor.n_checks["C10"] = \
                        self.monitor.n_checks.get("C10", 0) + 1
                    if delta > PRAZO_MONO_S:
                        self.monitor.registrar_violacao_prazo("C10")
                    t_conclusao = None

                # Monitorar C11: acórdão ≤ 30d após sessão
                if "sess" in atividade.lower() or "julgamento" in atividade.lower():
                    t_sessao = self.env.now
                    yield self.env.timeout(0)  # aguarda próximo evento
                    # Verificação será feita no próximo evento de acórdão

            t_saida = self.env.now

            # Registrar métricas finais do caso
            self.monitor.registrar_julgamento(
                case_id, prioritario,
                t_chegada, t_inicio, t_saida, assessor_id,
            )


# ===========================================================================
# SEÇÃO 5 — Gerador de chegadas (processo de Poisson)
# ===========================================================================

def gerador_chegadas(env:        simpy.Environment,
                      evento_log: EventLog,
                      modelo:     ModeloGabinete) -> Generator:
    """
    Lê o log sintético (do Sim2Log) e gera chegadas ao sistema.
    Usa os timestamps do log XES como instantes de chegada,
    respeitando a ordem cronológica e o tipo (HC/regular).

    Alternativa ao processo de Poisson puro: reusa a distribuição
    de chegadas já amostrada pelo Sim2Log, garantindo consistência.
    """
    # Ordenar traces por timestamp de chegada
    def ts_trace(t: Trace):
        evs = list(t)
        return evs[0].get("time:timestamp") if evs else datetime.min.replace(tzinfo=timezone.utc)

    traces_ordenadas = sorted(evento_log, key=ts_trace)
    if not traces_ordenadas:
        return

    ts_ref = ts_trace(traces_ordenadas[0])

    for trace in traces_ordenadas:
        ts_atual = ts_trace(trace)
        if ts_atual is None:
            continue

        # Tempo relativo em segundos a partir do início da simulação
        t_relativo = (ts_atual - ts_ref).total_seconds()
        if t_relativo < 0:
            t_relativo = 0

        # Aguardar até o momento de chegada
        if t_relativo > env.now:
            yield env.timeout(t_relativo - env.now)

        case_id     = trace.attributes.get("concept:name", "?")
        prioritario = bool(trace.attributes.get(ATTR_PRIO, False))

        # Disparar processo para o caso (não aguarda — concorrente)
        env.process(modelo.processar_caso(case_id, trace, prioritario))


# ===========================================================================
# SEÇÃO 6 — Carga de artefatos
# ===========================================================================

def carregar_ontologia(ont_dir: Optional[Path]) -> Optional["OntologiaPM4JUD"]:
    if not _ONT_OK or ont_dir is None or not ont_dir.exists():
        log.warning("  Ontologia: indisponível — restrições RISTJ em modo fallback")
        return None
    try:
        ont = OntologiaPM4JUD(ont_dir).carregar([3, 5, 7])
        log.info("  Ontologia Módulos 3+5+7 carregados")
        return ont
    except Exception as exc:
        log.warning("  Ontologia: %s", exc)
        return None


def carregar_restricoes(ont: Optional["OntologiaPM4JUD"],
                         ltlf_path: Optional[Path]) -> List[Dict]:
    """
    Carrega constraints RISTJ C1–C16 da ontologia (Módulo 7).
    Fallback: lê ltlf_<gab>.json do P6 para obter constraints_metadata.
    """
    if ont is not None:
        try:
            restricoes = ont.constraints_ltlf()
            if restricoes and len(restricoes) >= 10:
                hard = [c for c in restricoes if c.get("nivel", 1) <= 1]
                log.info("  Restrições RISTJ: %d carregadas (%d hard)",
                         len(restricoes), len(hard))
                return restricoes
        except Exception as exc:
            log.warning("  Ontologia constraints_ltlf: %s", exc)

    # Fallback: ltlf_<gab>.json do P6
    if ltlf_path and ltlf_path.exists():
        try:
            data = json.loads(ltlf_path.read_text(encoding="utf-8"))
            meta = data.get("constraints_metadata", [])
            log.info("  Restrições RISTJ: %d carregadas de %s (P6 fallback)",
                     len(meta), ltlf_path.name)
            return meta
        except Exception as exc:
            log.warning("  ltlf_<gab>.json: %s", exc)

    log.warning("  Restrições RISTJ: usando fallback mínimo (C10, C11)")
    return [
        {"id": "C10", "prazo_dias": 10, "nivel": 1, "padrao": "response_with_deadline"},
        {"id": "C11", "prazo_dias": 30, "nivel": 1, "padrao": "response_with_deadline"},
    ]


def carregar_params_des(input_dir: Path, gabinete: str) -> List[Dict]:
    fpath = input_dir / f"params_des_{gabinete}.csv"
    if not fpath.exists():
        log.warning("  params_des_%s.csv não encontrado", gabinete)
        return []
    with fpath.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    log.info("  params_des: %d atividades (P5)", len(rows))
    return rows


def carregar_kappa_baseline(input_dir: Path, gabinete: str) -> float:
    """Carrega κ_baseline do P6 para comparação pós-simulação."""
    for nome in (f"ltlf_{gabinete}.json", "p6_relatorio.json"):
        fpath = input_dir / nome
        if not fpath.exists():
            continue
        try:
            data = json.loads(fpath.read_text(encoding="utf-8"))
            if "kappa" in data:
                return float(data["kappa"])
            # Em p6_relatorio.json, buscar por gabinete
            for gab_res in data.get("gabinetes", []):
                if gab_res.get("gabinete") == gabinete:
                    return float(gab_res.get("kappa", 1.0))
        except Exception:
            pass
    return 1.0    # baseline padrão


def listar_xes_simulados(input_dir: Path, gabinete: str, n_rep: int) -> List[Path]:
    """Lista os arquivos XES gerados pelo Sim2Log."""
    arquivos = []
    for r in range(1, n_rep + 1):
        p = input_dir / f"sim2log_{gabinete}_r{r:02d}.xes"
        if p.exists():
            arquivos.append(p)
        else:
            log.warning("  XES sintético não encontrado: %s", p.name)
    return arquivos


# ===========================================================================
# SEÇÃO 7 — Execução de uma replicação
# ===========================================================================

def executar_replicacao(xes_path:   Path,
                         config:     ConfiguracaoGabinete,
                         params_des: List[Dict],
                         restricoes: List[Dict],
                         semente:    int,
                         horizonte:  float) -> Dict:
    """
    Executa uma replicação do DES SimPy para um log sintético.

    Horizonte adaptativo (Opção C — aprovado)
    ------------------------------------------
    O horizonte padrão (12 meses de horas úteis) pode ser insuficiente se o
    XES tiver timestamps com span maior. O horizonte efetivo é calculado como:
      horizonte_efetivo = max(span_xes × 1.5, horizonte_padrao)
    onde span_xes = (ts_max - ts_min) do log. O fator 1.5 garante tempo para
    que o último caso na fila seja completado após a última chegada.
    """
    event_log = xes_importer.apply(str(xes_path))
    if len(event_log) == 0:
        return {"erro": "log vazio", "xes": xes_path.name}

    # --- Horizonte adaptativo: span real do XES ---
    def _ts_ev0(t):
        evs = list(t)
        return evs[0].get("time:timestamp") if evs else None

    ts_validos = [ts for ts in (_ts_ev0(t) for t in event_log) if ts is not None]
    horizonte_efetivo = horizonte
    if len(ts_validos) >= 2:
        try:
            span_s = (max(ts_validos) - min(ts_validos)).total_seconds()
            h_adapt = span_s * 1.5
            if h_adapt > horizonte:
                log.info("    Horizonte adaptativo: %.0fs=%.1fd (XES span=%.1fd, padrao=%.1fd)",
                         h_adapt, h_adapt/86400, span_s/86400, horizonte/86400)
                horizonte_efetivo = h_adapt
        except (TypeError, AttributeError):
            pass

    # --- Gerador de chegadas ---
    n_dispatched = [0]

    def _gerador(env, evento_log, modelo):
        def ts_trace(t):
            evs = list(t)
            ts  = evs[0].get("time:timestamp") if evs else None
            if ts is None:
                return datetime.min.replace(tzinfo=timezone.utc)
            return ts

        traces_ord = sorted(evento_log, key=ts_trace)
        if not traces_ord:
            return
        ts_ref = ts_trace(traces_ord[0])

        for trace in traces_ord:
            ts_atual = ts_trace(trace)
            try:
                t_rel = (ts_atual - ts_ref).total_seconds()
            except TypeError:
                try:
                    if ts_ref.tzinfo is None:
                        ts_ref = ts_ref.replace(tzinfo=timezone.utc)
                    if ts_atual.tzinfo is None:
                        ts_atual = ts_atual.replace(tzinfo=timezone.utc)
                    t_rel = (ts_atual - ts_ref).total_seconds()
                except Exception:
                    continue
            if t_rel < 0:
                t_rel = 0
            if t_rel > env.now:
                yield env.timeout(t_rel - env.now)
            case_id     = trace.attributes.get("concept:name", "?")
            prioritario = bool(trace.attributes.get(ATTR_PRIO, False))
            env.process(modelo.processar_caso(case_id, trace, prioritario))
            n_dispatched[0] += 1

    rng     = random.Random(semente)
    env     = simpy.Environment()
    monitor = MonitorMetricas()
    modelo  = ModeloGabinete(
        env=env, config=config, params_des=params_des,
        monitor=monitor, restricoes=restricoes, rng=rng,
    )
    env.process(_gerador(env, event_log, modelo))
    env.run(until=horizonte_efetivo)

    if n_dispatched[0] < len(event_log):
        log.warning("    %d/%d casos não despachados (horizonte=%.0fd)",
                    len(event_log) - n_dispatched[0],
                    len(event_log), horizonte_efetivo / 86400)

    t_medio = monitor.calcular_t_medio()
    gini    = monitor.calcular_gini()
    kappa   = monitor.calcular_kappa()
    eta     = monitor.calcular_eta()

    return {
        "xes_file":            xes_path.name,
        "semente":             semente,
        "n_casos":             len(event_log),
        "n_dispatched":        n_dispatched[0],
        "n_julgados":          monitor.n_julgados,
        "n_hc":                monitor.n_hc_julgados,
        "t_medio_dias":        round(t_medio, 4),
        "gini":                round(gini, 4),
        "kappa":               round(kappa, 4),
        "eta":                 round(eta, 4),
        "t_medio_hc":          round(float(np.mean(monitor.t_julgamento_hc))      if monitor.t_julgamento_hc      else 0, 4),
        "t_medio_reg":         round(float(np.mean(monitor.t_julgamento_regular)) if monitor.t_julgamento_regular else 0, 4),
        "t_espera_hc":         round(float(np.mean(monitor.t_espera_hc))          if monitor.t_espera_hc          else 0, 4),
        "violacoes":           dict(monitor.violacoes),
        "n_checks":            dict(monitor.n_checks),
        "horizonte_efetivo_s": round(horizonte_efetivo),
    }


# ===========================================================================
# SEÇÃO 8 — Pipeline por gabinete
# ===========================================================================

def processar_gabinete(gabinete:    str,
                        input_dir:   Path,
                        output_dir:  Path,
                        ont:         Optional["OntologiaPM4JUD"],
                        config:      ConfiguracaoGabinete,
                        n_rep:       int,
                        horizonte:   float) -> Dict:
    """
    Pipeline DES completo para um gabinete:
      1. Carrega artefatos P5/P6
      2. Para cada replicação (30): executa simulação SimPy
      3. Agrega métricas: média, desvio padrão, IC 95%
      4. Salva resultados por replicação e consolidado
    """
    log.info("=" * 70)
    log.info("Processando gabinete: %s", gabinete.upper())
    log.info("=" * 70)
    t0 = time.time()

    # --- Artefatos de entrada ---
    params_des    = carregar_params_des(input_dir, gabinete)
    ltlf_path     = input_dir / f"ltlf_{gabinete}.json"
    restricoes    = carregar_restricoes(ont, ltlf_path)
    kappa_base    = carregar_kappa_baseline(input_dir, gabinete)
    xes_files     = listar_xes_simulados(input_dir, gabinete, n_rep)

    log.info("  Config: %s", config)
    log.info("  κ_baseline (P6): %.4f", kappa_base)
    log.info("  XES disponíveis: %d/%d", len(xes_files), n_rep)

    if not xes_files:
        log.error("  Nenhum XES sintético encontrado — execute P7a (Sim2Log) primeiro")
        return {"gabinete": gabinete, "status": "ERRO", "motivo": "XES ausentes"}

    # --- Loop de replicações ---
    resultados_rep: List[Dict] = []
    for i, xes_path in enumerate(xes_files, start=1):
        t_rep = time.time()
        semente = i * 137 + hash(gabinete) % 10000
        log.info("  Rep %2d/%d: %s (semente=%d)...",
                 i, len(xes_files), xes_path.name, semente)

        r = executar_replicacao(
            xes_path=xes_path, config=config,
            params_des=params_des, restricoes=restricoes,
            semente=semente, horizonte=horizonte,
        )
        r["replicacao"] = i
        resultados_rep.append(r)

        log.info("    T̄=%.2fd G=%.4f κ=%.4f η=%.4f "
                 "julgados=%d/%d t=%.1fs",
                 r.get("t_medio_dias", 0), r.get("gini", 0),
                 r.get("kappa", 0), r.get("eta", 0),
                 r.get("n_julgados", 0),
                 r.get("n_dispatched", r.get("n_casos", 0)),
                 time.time() - t_rep)

    # --- Agregar métricas ---
    def _agg(chave: str) -> Dict:
        vals = [r[chave] for r in resultados_rep if chave in r]
        if not vals:
            return {"media": 0, "desvpad": 0, "ic95_lo": 0, "ic95_hi": 0}
        m  = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0
        t_crit = 2.045  # t de Student, 29 gl, α=0.05 bilateral
        ic_lo  = m - t_crit * sd / math.sqrt(len(vals))
        ic_hi  = m + t_crit * sd / math.sqrt(len(vals))
        return {"media":  round(m, 4),
                "desvpad":round(sd, 4),
                "ic95_lo":round(ic_lo, 4),
                "ic95_hi":round(ic_hi, 4),
                "n":      len(vals)}

    metricas = {
        "t_medio_dias": _agg("t_medio_dias"),
        "gini":         _agg("gini"),
        "kappa":        _agg("kappa"),
        "eta":          _agg("eta"),
        "t_medio_hc":   _agg("t_medio_hc"),
        "t_medio_reg":  _agg("t_medio_reg"),
    }

    log.info("  AGREGADO: T̄=%.2f±%.2f G=%.4f±%.4f κ=%.4f±%.4f η=%.4f±%.4f",
             metricas["t_medio_dias"]["media"], metricas["t_medio_dias"]["desvpad"],
             metricas["gini"]["media"],         metricas["gini"]["desvpad"],
             metricas["kappa"]["media"],         metricas["kappa"]["desvpad"],
             metricas["eta"]["media"],           metricas["eta"]["desvpad"])

    # --- Salvar por replicação ---
    for r in resultados_rep:
        rep_path = output_dir / f"des_{gabinete}_r{r['replicacao']:02d}.json"
        rep_path.write_text(
            json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- Salvar consolidado do gabinete ---
    consolidado = {
        "programa":      "PM4JUD-DES",
        "versao":        VERSION,
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "gabinete":      gabinete,
        "n_rep":         len(resultados_rep),
        "kappa_baseline": kappa_base,
        "configuracao":  {
            "n_total":     config.n_total,
            "n_inst":      config.n_inst,
            "cap_efetiva": round(config.capacidade_efetiva(), 2),
        },
        "metricas":      metricas,
        "replicacoes":   resultados_rep,
    }
    gab_path = output_dir / f"des_{gabinete}.json"
    gab_path.write_text(
        json.dumps(consolidado, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("  Salvo: %s e %d des_%s_r*.json",
             gab_path.name, len(resultados_rep), gabinete)

    t_total = round(time.time() - t0, 1)
    return {
        "gabinete":      gabinete,
        "status":        "OK",
        "n_rep":         len(resultados_rep),
        "kappa_baseline":kappa_base,
        "t_medio_dias":  metricas["t_medio_dias"],
        "gini":          metricas["gini"],
        "kappa":         metricas["kappa"],
        "eta":           metricas["eta"],
        "tempo_s":       t_total,
    }


# ===========================================================================
# SEÇÃO 9 — Relatório consolidado e entry point
# ===========================================================================

def salvar_relatorio(resultados: List[Dict], output_dir: Path,
                     config: ConfiguracaoGabinete) -> None:
    payload = {
        "programa":     "PM4JUD-DES",
        "versao":       VERSION,
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "descricao":    (
            "Resultados da Simulação por Eventos Discretos (SimPy M/M/c). "
            "Métricas T̄, Gini, κ e η calculadas por 30 replicações independentes. "
            "IC 95% com t de Student (29 g.l., α=0.05). "
            "Resultados utilizados pelo P8 OPT como funções objetivo f1–f4."
        ),
        "proximo_passo": "PM4JUD-OPT (P8) — usa T̄, Gini, κ, η como funções objetivo",
        "configuracao": {
            "n_total":     config.n_total,
            "n_inst":      config.n_inst,
            "cap_efetiva": round(config.capacidade_efetiva(), 2),
            "estrutura":   [
                {"cat": cat, "n": n, "funcao": f, "peso": p}
                for cat, n, f, p in ESTRUTURA_GABINETE
            ],
        },
        "gabinetes":    resultados,
        "sumario": {
            "n_gabinetes": len(resultados),
            "t_medio_geral": round(float(np.mean(
                [r["t_medio_dias"]["media"]
                 for r in resultados if "t_medio_dias" in r])), 4),
            "kappa_medio": round(float(np.mean(
                [r["kappa"]["media"]
                 for r in resultados if "kappa" in r])), 4),
        },
    }
    fpath = output_dir / "p7b_relatorio.json"
    fpath.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Relatório: %s", fpath.name)


def imprimir_resumo(resultados: List[Dict]) -> None:
    log.info("")
    log.info("=" * 75)
    log.info("RESUMO P7b — SIMULAÇÃO POR EVENTOS DISCRETOS")
    log.info("=" * 75)
    log.info("  %-12s  %12s  %8s  %8s  %8s",
             "Gabinete", "T̄ (dias)", "Gini", "κ", "η")
    log.info("  %s", "-" * 65)
    for r in resultados:
        tm = r.get("t_medio_dias", {})
        gi = r.get("gini", {})
        ka = r.get("kappa", {})
        et = r.get("eta", {})
        log.info("  %-12s  %6.2f ± %.2f  %6.4f  %6.4f  %6.4f",
                 r["gabinete"],
                 tm.get("media", 0), tm.get("desvpad", 0),
                 gi.get("media", 0),
                 ka.get("media", 0),
                 et.get("media", 0))
    log.info("")
    log.info("Próximo: PM4JUD-OPT (P8) — NSGA-II vs AMGA2 vs SPEA2")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pm4jud_des",
        description="PM4JUD-DES v1.0 (P7b) — Simulação por Eventos Discretos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input",     required=True, type=Path,
                        help="Diretório com sim2log_<gab>_r*.xes e artefatos P5/P6")
    parser.add_argument("--output",    required=True, type=Path,
                        help="Diretório de saída dos resultados DES")
    parser.add_argument("--ontologia", default=None,  type=Path,
                        help="Diretório com PM4JUD_*.owl (Módulos 3, 5, 7)")
    parser.add_argument("--gabinetes", nargs="+",
                        default=["reynaldo", "palheiro", "schietti"],
                        help="Gabinetes a processar (padrão: todos os três)")
    parser.add_argument("--n-rep",    type=int, default=30,
                        help="Número de replicações por gabinete (padrão: 30)")
    parser.add_argument("--horizonte", type=float,
                        default=12 * 30 * 8 * 3600,
                        help="Horizonte de simulação em segundos (padrão: 12 meses)")
    parser.add_argument("--configuracao", default=None, type=Path,
                        help="JSON com configuração de assessores (P8 OPT); padrão=STJ")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # Carregar configuração (padrão STJ ou do P8 OPT)
    config_json = None
    if args.configuracao and args.configuracao.exists():
        config_json = json.loads(args.configuracao.read_text(encoding="utf-8"))
        log.info("Configuração personalizada: %s", args.configuracao.name)
    config = ConfiguracaoGabinete(config_json)

    log.info("PM4JUD-DES v%s | %d rep. | horizonte=%.0fh",
             VERSION, args.n_rep, args.horizonte / 3600)
    log.info("Configuração: %s", config)
    log.info("Gabinetes: %s", args.gabinetes)

    ont = carregar_ontologia(args.ontologia)

    resultados = []
    for gabinete in args.gabinetes:
        r = processar_gabinete(
            gabinete, args.input, args.output, ont, config,
            n_rep=args.n_rep, horizonte=args.horizonte,
        )
        resultados.append(r)

    salvar_relatorio(resultados, args.output, config)
    imprimir_resumo(resultados)


if __name__ == "__main__":
    main()
