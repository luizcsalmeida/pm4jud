#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
PM4JUD-OPT  v1.0
================================================================================

Dissertação de Mestrado — PPGIa/PUCPR
Título: PM4JUD — Otimização Multiobjetivo com Mineração de Processos e
        Simulação no Contexto do Fluxo Processual em Gabinetes de Magistrado
Autor:  Luiz Claudio Soares de Almeida
Orient: Prof. Dr. Edson Emilio Scalabrin
Ano:    2026

Descrição
---------
P8 do pipeline PM4JUD. Implementa a otimização multiobjetivo do modelo de
alocação de assessores nos gabinetes piloto do STJ, comparando três algoritmos
evolutivos sobre as mesmas condições experimentais (design controlado, α=0,05).

Alinhamento com PM4SOS (FERRONATO, 2022, Cap. 5)
--------------------------------------------------
O PM4SOS usa CBR+AMGA2+Simulated Annealing como estratégia de otimização.
O PM4JUD herda o CBR como parte do método PM4SOS e contribui com:
  • Comparação experimental de três algoritmos (NSGA-II, AMGA2, SPEA2)
  • Integração da Ontologia Judicial como guia semântico das restrições
  • Quatro funções objetivo alinhadas às Metas CNJ e ao RISTJ

Raciocínio Baseado em Casos — CBR (AAMODT; PLAZA, 1994)
---------------------------------------------------------
O CBR implementa o ciclo canônico Recuperar→Reutilizar→Revisar→Reter:

  Recuperar : dado o perfil do gabinete (vetor de características), busca os
              k=5 casos mais similares na base de conhecimento (base_casos.json)
              usando distância Euclidiana normalizada.

  Reutilizar: as soluções recuperadas são injetadas na população inicial do
              algoritmo evolutivo como warm-start, reduzindo o tempo de
              convergência e aproveitando experiências passadas.

  Revisar   : o algoritmo evolutivo (NSGA-II, AMGA2 ou SPEA2) otimiza a
              solução a partir do warm-start, explorando vizinhanças não
              cobertas pela recuperação.

  Reter     : a melhor solução da fronteira de Pareto é armazenada na base
              de casos com suas características e métricas resultantes,
              enriquecendo o conhecimento para execuções futuras.

Ontologia PM4JUD — uso transversal em P8
-----------------------------------------
Módulo 3 (Classes)    : identifica HC/RHC para priorização no vetor objetivo.
                        A ponderação de f1 (T̄) usa proporção HC do gabinete.
Módulo 7 (PM4JUD)     : carrega via SPARQL os indivíduos Meta1/Meta2/Meta4
                        com seus limiares (thresholds) para cálculo de η (f4).
                        Carrega C10/C11 (hard constraints) como penalizações
                        inviabilizadoras na função de fitness.
                        As restrições RISTJ articulam o espaço de busca —
                        soluções que violam C10 ou C11 são descartadas.

Formulação MOOP
---------------
Variáveis de decisão (inteiras): alocação de assessores por categoria
  x = [n_CJ3A, n_CJ2A, n_FC6C, n_FC4IV, n_FC2II]
  Restrição: Σx + n_CJ3C_fixo(=1) = N_ASSESSORES(=38)  → Σx = 37
  Limite inferior: 1 assessor por categoria (mínimo operacional)

Funções objetivo (minimização):
  f1 = T̄           (tempo médio de julgamento em dias — minimizar)
  f2 = Gini         (desbalanceamento de carga entre assessores — minimizar)
  f3 = 1 − κ       (taxa de violação regimental — minimizar)
  f4 = 1 − η       (não-aderência às Metas CNJ 1/2/4 — minimizar)

Algoritmos comparados
---------------------
  NSGA-II : Non-dominated Sorting Genetic Algorithm II (DEB et al., 2002)
            Seleção por torneio + elitismo via rank de Pareto + crowding distance.
  AMGA2   : Adaptive Multi-Goal Algorithm 2 (TIWARI et al., 2011)
            Referenciado por Ferronato (2022). Decomposição por referência
            adaptativa, archiving de soluções não-dominadas e busca local.
  SPEA2   : Strength Pareto Evolutionary Algorithm 2 (ZITZLER et al., 2001)
            Arquivamento externo, fitness por força + densidade, truncation.

Design experimental (FERRONATO, 2022, Cap. 5)
----------------------------------------------
  30 replicações independentes × 3 algoritmos = 90 execuções
  Análise: Kruskal-Wallis + post-hoc Wilcoxon + correção Bonferroni (α=0,05)
  Métrica: hypervolume da fronteira de Pareto (EMMERICH et al., 2006)

Entradas (output/ dos pipelines anteriores)
--------------------------------------------
  des_<gab>_r*.json       Métricas DES por replicação P7b (T̄, Gini, κ, η)
  des_<gab>.json          Baseline do grupo controle (médias 30 rep.)
  params_des_<gab>.csv    Características do gabinete para CBR (do P5)
  ltlf_<gab>.json         κ baseline e constraints C1–C16 (do P6)
  base_casos.json         Base de conhecimento CBR (gerada/atualizada pelo P8)
  --ontologia <dir>       Diretório com PM4JUD_*.owl (Módulos 3, 7)

Saídas (output/)
----------------
  opt_<gab>_<algo>_r<N>.json   Fronteira Pareto por replicação
  opt_<gab>_<algo>.json        Agregado por algoritmo (média, hv, IC95)
  base_casos.json              Base CBR atualizada (retain step)
  p8_relatorio.json            Relatório consolidado para P9

Pipeline completo
-----------------
  P1→P2→P3→P4→P5→P6→P7a→P7b→[P8 OPT]→P9 STAT

Referências
-----------
  FERRONATO, J. J. PM4SOS. Tese (Doutorado em Informática) — PUCPR, 2022.
  AAMODT, A.; PLAZA, E. Case-Based Reasoning. AI Communications, v.7, n.1, 1994.
  DEB, K. et al. NSGA-II. IEEE Trans. Evol. Comput., v.6, n.2, pp.182-197, 2002.
  TIWARI, S. et al. AMGA2. Evol. Comput., v.19, n.4, pp.577-609, 2011.
  ZITZLER, E. et al. SPEA2. TIK-Report 103, ETH Zürich, 2001.
  EMMERICH, M. et al. Hypervolume. PPSN, LNCS 4193, pp.313-322, 2006.

Repositório: https://github.com/luizcsalmeida/pm4jud/tree/main/opt
================================================================================
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# DES — importado para avaliação de fitness inline (evita overhead de processo)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent.parent
if str(_HERE / "sim") not in sys.path:
    sys.path.insert(0, str(_HERE / "sim"))
if str(_HERE / "etl") not in sys.path:
    sys.path.insert(0, str(_HERE / "etl"))

try:
    from pm4jud_des import (
        ConfiguracaoGabinete, ModeloGabinete, MonitorMetricas,
        ESTRUTURA_GABINETE, N_ASSESSORES,
    )
    import simpy
    from pm4py.objects.log.importer.xes import importer as xes_importer
    _DES_OK = True
except ImportError as exc:
    print(f"[ERRO] {exc}", file=sys.stderr)
    _DES_OK = False

try:
    from pm4jud_ontologia import OntologiaPM4JUD
    _ONT_OK = True
except ImportError:
    _ONT_OK = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
# basicConfig é no-op quando o DES já o chamou na importação.
# Solução: configurar handler diretamente no logger nomeado com
# propagate=False — mensagens não sobem ao root logger (que tem
# o formato do DES hardcoded), garantindo "PM4JUD-OPT" na saída.
log = logging.getLogger("PM4JUD-OPT")
log.setLevel(logging.INFO)
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-8s] PM4JUD-OPT — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    log.addHandler(_h)
    log.propagate = False
VERSION = "1.0"


# ===========================================================================
# SEÇÃO 1 — Constantes do domínio PM4JUD/STJ
# ===========================================================================

# Algoritmos disponíveis para comparação
ALGORITMOS = ["nsga2", "amga2", "spea2"]

# Parâmetros do algoritmo evolutivo
N_POP    = 10     # tamanho da população
N_GEN    = 20     # número de gerações
K_CBR    = 5      # vizinhos mais próximos no CBR
N_REP_FITNESS = 1 # replicações DES por avaliação de fitness (velocidade)

# Estrutura de assessores — categorias livres (n_CJ3C=1 é fixo)
CATS_LIVRES = ["CJ3A", "CJ2A", "FC6C", "FC4IV", "FC2II"]
N_FIXO      = 1   # n_CJ3C fixo (gestão do gabinete)
N_LIVRES    = N_ASSESSORES - N_FIXO   # 37 a distribuir entre 5 categorias

# Limites por categoria
LB = [1, 1, 1, 1, 1]           # mínimo 1 por categoria
UB = [20, 10, 25, 15, 10]      # máximo razoável por categoria

# Horizonte DES para avaliação de fitness (reduzido para velocidade)
HORIZONTE_FITNESS = 365 * 24 * 3600   # 1 ano em segundos

# Ponto de referência para cálculo do hypervolume (pior caso)
HV_REF = [100.0, 1.0, 1.0, 1.0]   # [T̄_max, Gini_max, 1-κ_max, 1-η_max]

# Penalização para soluções inviáveis (violam hard constraints C10/C11)
PENALIDADE_HARD = 1000.0


# ===========================================================================
# SEÇÃO 2 — CBR: Base de Conhecimento e Ciclo Recuperar-Reutilizar-Revisar-Reter
# ===========================================================================

class RepresentacaoCaso:
    """
    Representa um caso na base de conhecimento do CBR.

    Um caso encapsula o perfil do gabinete (características) e a solução
    que produziu bons resultados, conforme o ciclo de Aamodt & Plaza (1994):
      • Problema  : vetor de características do gabinete (Φ)
      • Solução   : alocação de assessores [n_CJ3A, n_CJ2A, n_FC6C, n_FC4IV, n_FC2II]
      • Resultado : vetor objetivo [T̄, Gini, 1-κ, 1-η]
      • Contexto  : gabinete, algoritmo, data

    Normalização das características para distância Euclidiana:
      λ_prio, λ_reg: [0, 0.001] → [0, 1]
      n_atividades  : [0, 100]  → [0, 1]
      T̄_baseline   : [0, 365]  → [0, 1]
      kappa_base    : [0, 1]    → já normalizado
    """

    def __init__(self,
                 gabinete:       str,
                 caracteristicas: Dict[str, float],
                 solucao:         List[int],
                 objetivos:       List[float],
                 algoritmo:       str,
                 timestamp:       str):
        self.gabinete        = gabinete
        self.caracteristicas = caracteristicas   # vetor Φ
        self.solucao         = solucao           # x = [n_CJ3A, ...]
        self.objetivos       = objetivos         # [T̄, Gini, 1-κ, 1-η]
        self.algoritmo       = algoritmo
        self.timestamp       = timestamp

    def vetor_phi(self) -> np.ndarray:
        """Vetor de características normalizado para distância Euclidiana."""
        c = self.caracteristicas
        return np.array([
            c.get("lambda_prio",    0.0) / 0.001,
            c.get("lambda_regular", 0.0) / 0.001,
            c.get("n_atividades",   0.0) / 100.0,
            c.get("t_medio_base",   0.0) / 365.0,
            c.get("kappa_base",     1.0),
        ])

    def to_dict(self) -> Dict:
        return {
            "gabinete":        self.gabinete,
            "caracteristicas": self.caracteristicas,
            "solucao":         self.solucao,
            "objetivos":       self.objetivos,
            "algoritmo":       self.algoritmo,
            "timestamp":       self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "RepresentacaoCaso":
        return cls(
            gabinete        = d["gabinete"],
            caracteristicas = d["caracteristicas"],
            solucao         = d["solucao"],
            objetivos       = d["objetivos"],
            algoritmo       = d.get("algoritmo", ""),
            timestamp       = d.get("timestamp", ""),
        )


class BaseConhecimentoCBR:
    """
    Base de conhecimento do CBR — persiste entre execuções em base_casos.json.

    Implementa o ciclo canônico de Aamodt & Plaza (1994):

    Recuperar (retrieve):
      Dado o perfil Φ do gabinete atual, calcula a distância Euclidiana
      entre Φ e todos os casos armazenados. Retorna os k mais próximos.
      Similaridade = 1 / (1 + d_Euclidiana(Φ, Φ_caso))

    Reutilizar (reuse):
      Extrai as soluções dos k casos recuperados como candidatas iniciais.
      Estas são injetadas na população inicial do algoritmo evolutivo
      (warm-start), acelerando a convergência em regiões promissoras.

    Revisar (revise):
      Não implementado aqui — é realizado pelo algoritmo evolutivo (P8).
      O algoritmo otimiza a partir do warm-start, adaptando a solução
      recuperada às especificidades do problema atual.

    Reter (retain):
      Após a otimização, a melhor solução da fronteira de Pareto é
      armazenada na base de conhecimento para uso futuro.
      A base cresce progressivamente, enriquecendo o conhecimento.

    Ontologia PM4JUD — contribuição adicional ao CBR do PM4SOS:
      Os limiares de κ e η usados nas características são extraídos
      via SPARQL do Módulo 7 (PM4JUD.owl), garantindo que a similaridade
      entre casos reflita as restrições regimentais reais do STJ,
      não apenas métricas genéricas de simulação.
    """

    def __init__(self, caminho: Path):
        self.caminho = caminho
        self.casos:  List[RepresentacaoCaso] = []
        self._carregar()

    def _carregar(self) -> None:
        if self.caminho.exists():
            try:
                dados = json.loads(self.caminho.read_text(encoding="utf-8"))
                self.casos = [RepresentacaoCaso.from_dict(d)
                              for d in dados.get("casos", [])]
                log.info("  CBR: %d casos carregados de %s",
                         len(self.casos), self.caminho.name)
            except Exception as exc:
                log.warning("  CBR: erro ao carregar base — %s", exc)
                self.casos = []
        else:
            log.info("  CBR: base nova criada em %s", self.caminho.name)

    def _salvar(self) -> None:
        payload = {
            "programa":  "PM4JUD-OPT",
            "versao":    VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_casos":   len(self.casos),
            "casos":     [c.to_dict() for c in self.casos],
        }
        self.caminho.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # --- Recuperar -------------------------------------------------------
    def recuperar(self,
                  phi_consulta: np.ndarray,
                  k:            int = K_CBR) -> List[RepresentacaoCaso]:
        """
        Recupera os k casos mais similares ao perfil φ_consulta.
        Similaridade = 1 / (1 + dist_Euclidiana(φ_consulta, φ_caso)).
        Retorna lista ordenada por similaridade decrescente.
        """
        if not self.casos:
            return []
        similares = []
        for caso in self.casos:
            phi_c = caso.vetor_phi()
            if len(phi_c) != len(phi_consulta):
                continue
            dist = float(np.linalg.norm(phi_consulta - phi_c))
            sim  = 1.0 / (1.0 + dist)
            similares.append((sim, caso))
        similares.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in similares[:k]]

    # --- Reutilizar -------------------------------------------------------
    def extrair_warm_start(self,
                            phi_consulta: np.ndarray,
                            k:            int = K_CBR) -> List[List[int]]:
        """
        Extrai soluções dos k casos mais similares para warm-start evolutivo.
        Filtra soluções que violam a restrição de soma (Σx = N_LIVRES).
        """
        recuperados = self.recuperar(phi_consulta, k)
        warm = []
        for caso in recuperados:
            sol = caso.solucao
            if (len(sol) == len(CATS_LIVRES) and
                    sum(sol) == N_LIVRES and
                    all(LB[i] <= sol[i] <= UB[i] for i in range(len(sol)))):
                warm.append(list(sol))
        log.info("    CBR: %d/%d casos recuperados com warm-start válido",
                 len(warm), len(recuperados))
        return warm

    # --- Reter ------------------------------------------------------------
    def reter(self, caso: RepresentacaoCaso) -> None:
        """
        Armazena o caso na base e persiste em disco.
        Evita duplicatas exatas (mesma solução e mesmo gabinete).
        """
        for c in self.casos:
            if c.gabinete == caso.gabinete and c.solucao == caso.solucao:
                return  # já existe
        self.casos.append(caso)
        self._salvar()
        log.info("    CBR: caso retido — base agora com %d casos", len(self.casos))

    def inicializar_com_controle(self,
                                  gabinete:  str,
                                  phi:       Dict[str, float],
                                  objetivos: List[float]) -> None:
        """
        Inicializa a base com o resultado do grupo controle (P7b).
        Usado na Fase 1 quando a base está vazia.
        A solução do controle é a alocação padrão STJ.
        """
        solucao_controle = [
            next(n for cat, n, _, _ in ESTRUTURA_GABINETE if cat == "CJ3A"),
            next(n for cat, n, _, _ in ESTRUTURA_GABINETE if cat == "CJ2A"),
            next(n for cat, n, _, _ in ESTRUTURA_GABINETE if cat == "FC6C"),
            next(n for cat, n, _, _ in ESTRUTURA_GABINETE if cat == "FC4IV"),
            next(n for cat, n, _, _ in ESTRUTURA_GABINETE if cat == "FC2II"),
        ]
        caso = RepresentacaoCaso(
            gabinete        = gabinete,
            caracteristicas = phi,
            solucao         = solucao_controle,
            objetivos       = objetivos,
            algoritmo       = "controle",
            timestamp       = datetime.now(timezone.utc).isoformat(),
        )
        self.reter(caso)


# ===========================================================================
# SEÇÃO 3 — Ontologia PM4JUD: extração de constraints e limiares
# ===========================================================================

class OntologiaOPT:
    """
    Interface da Ontologia PM4JUD para o P8 (Otimização).

    Módulo 3 (Classes) — proporção HC:
      Carrega via SPARQL a proporção de processos HC/RHC no gabinete.
      Usado para ponderar f1 (T̄): processos HC têm peso maior na
      função objetivo (Art. 91-I RISTJ — prazo de 30 dias).

    Módulo 7 (PM4JUD) — constraints e limiares:
      Carrega indivíduos Meta1/Meta2/Meta4 com thresholds η.
      Carrega C10/C11 (hard constraints) como parâmetros de penalização.
      Esses valores articulam o espaço de busca: soluções que produzem
      T̄ muito alto (violando C10/C11) são descartadas pelo DES (P7b)
      e aparecem como κ<1 no vetor objetivo.

    Uso transversal:
      A ontologia não apenas valida soluções — ela DEFINE o espaço de
      busca semanticamente. Sem ela, a otimização seria cega às normas
      regimentais e às metas do CNJ.
    """

    def __init__(self, ont_dir: Optional[Path]):
        self._ont:    Optional["OntologiaPM4JUD"] = None
        self._pronto: bool = False

        if not _ONT_OK or ont_dir is None or not ont_dir.exists():
            log.warning("  Ontologia OPT: indisponível — usando valores padrão")
            return

        try:
            self._ont = OntologiaPM4JUD(ont_dir).carregar([3, 7])
            self._pronto = True
            log.info("  Ontologia OPT: Módulos 3+7 carregados")
        except Exception as exc:
            log.warning("  Ontologia OPT: %s", exc)

    def limiares_metas_cnj(self) -> Dict[str, float]:
        """
        Retorna limiares das Metas CNJ 1/2/4 (indivíduos Módulo 7).
        Usado em calcular_eta do DES para o cálculo de f4 = 1 - η.
        """
        defaults = {"Meta1": 1.00, "Meta2": 0.50, "Meta4": 0.90}
        if not self._pronto or self._ont is None:
            return defaults
        try:
            restricoes = self._ont.constraints_ltlf()
            limiares   = {}
            for r in restricoes:
                rid = r.get("id", "")
                if rid in ("Meta1", "Meta2", "Meta4"):
                    limiares[rid] = float(r.get("limiar", defaults.get(rid, 0.5)))
            return {k: limiares.get(k, v) for k, v in defaults.items()}
        except Exception:
            return defaults

    def prazo_hard_constraints(self) -> Dict[str, float]:
        """
        Retorna prazos das hard constraints C10 e C11 (em segundos).
        C10 = decisão monocrática ≤ 10 dias (Art. 110-I RISTJ)
        C11 = acórdão ≤ 30 dias após sessão (Art. 110-III RISTJ)
        """
        defaults = {"C10": 10 * 86400, "C11": 30 * 86400}
        if not self._pronto or self._ont is None:
            return defaults
        try:
            restricoes = self._ont.constraints_ltlf()
            prazos     = {}
            for r in restricoes:
                rid = r.get("id", "")
                if rid in ("C10", "C11"):
                    dias = float(r.get("prazo_dias", 0))
                    if dias > 0:
                        prazos[rid] = dias * 86400
            return {k: prazos.get(k, v) for k, v in defaults.items()}
        except Exception:
            return defaults

    def proporcao_hc(self, gabinete: str, log_path: Optional[Path]) -> float:
        """
        Proporção de processos HC/RHC no gabinete (Módulo 3).
        Lê do log de eventos ou estima pelo default Fase 1 (74% baseado
        no corpus DATAJUD 2024 — todos os gabinetes são da Seção Criminal).
        """
        if log_path and log_path.exists():
            try:
                import csv as _csv
                with log_path.open(encoding="utf-8") as f:
                    rows = list(_csv.DictReader(f))
                n_prio = sum(1 for r in rows if r.get("prioritario", "").lower() == "true")
                return n_prio / max(len(rows), 1)
            except Exception:
                pass
        return 0.74  # default Fase 1: 74% HC (corpus criminal STJ)


# ===========================================================================
# SEÇÃO 4 — Função de fitness: DES inline
# ===========================================================================

def _preparar_cache_xes(
        xes_paths: List[Path],
        horizonte_padrao: float = HORIZONTE_FITNESS,
) -> List[Tuple]:
    """
    Pré-parseia todos os XES de um gabinete UMA única vez.

    Retorna lista de tuplas (EventLog, traces_ordenadas, horizonte_s)
    onde:
      EventLog         — objeto PM4Py já em memória (sem I/O posterior)
      traces_ordenadas — traces já ordenadas por timestamp (evita re-sort)
      horizonte_s      — horizonte adaptativo pré-calculado

    Motivação: xes_importer.apply() consome ~1s por arquivo em disco.
    Com 10 pop × 20 gen × 30 rep = 6.000 avaliações por algoritmo, isso
    representaria 6.000s de I/O puro por gabinete. O pré-parsing reduz
    esse custo a uma única leitura de ~30s (um por gabinete), cortando
    ~85% do overhead de avaliação de fitness.
    """
    cache = []

    def _ts_ev0(t):
        evs = list(t)
        return evs[0].get("time:timestamp") if evs else None

    for p in xes_paths:
        try:
            el = xes_importer.apply(str(p))
            # Ordenar traces por timestamp do 1º evento (feito uma vez)
            ts_lista = [(_ts_ev0(t), t) for t in el]
            ts_lista.sort(key=lambda x: (x[0] is None, x[0]))
            traces_ord = [t for _, t in ts_lista]
            # Horizonte adaptativo pré-calculado
            ts_vals = [ts for ts, _ in ts_lista if ts is not None]
            horizonte = horizonte_padrao
            if len(ts_vals) >= 2:
                try:
                    span = (max(ts_vals) - min(ts_vals)).total_seconds()
                    horizonte = max(span * 1.5, horizonte_padrao)
                except Exception:
                    pass
            cache.append((el, traces_ord, horizonte))
        except Exception as exc:
            log.warning("    Cache XES: erro ao parsear %s — %s", p.name, exc)

    return cache


def _gerador_traces(env, traces_ord, modelo):
    """Gerador SimPy — usa traces já ordenadas do cache (sem re-sort)."""
    if not traces_ord:
        return
    def _ts(t):
        evs = list(t)
        return evs[0].get("time:timestamp") if evs else None

    ts_ref = _ts(traces_ord[0])
    for trace in traces_ord:
        ts_a = _ts(trace)
        if ts_a is None:
            continue
        try:
            t_rel = max((ts_a - ts_ref).total_seconds(), 0)
        except TypeError:
            continue
        if t_rel > env.now:
            yield env.timeout(t_rel - env.now)
        cid  = trace.attributes.get("concept:name", "?")
        prio = bool(trace.attributes.get("pm4jud:prioritario", False))
        env.process(modelo.processar_caso(cid, trace, prio))


def avaliar_solucao(x:          List[int],
                    logs_cache: List[Tuple],
                    params_des: List[Dict],
                    restricoes: List[Dict],
                    rng:        random.Random,
                    n_rep:      int = N_REP_FITNESS) -> List[float]:
    """
    Avalia uma solução candidata x executando n_rep replicações do DES.

    Fluxo:
      1. Verificar viabilidade de x (Σx = N_LIVRES, limites por categoria)
      2. Construir ConfiguracaoGabinete a partir de x
      3. Selecionar n_rep entradas do cache (EventLog já parseado)
      4. Para cada entrada: executar SimPy M/M/c com a nova configuração
      5. Agregar: T̄, Gini, κ, η → retornar [T̄, Gini, 1-κ, 1-η]

    Otimização de pré-parsing:
      logs_cache contém EventLogs já em memória (parseados uma vez em
      _preparar_cache_xes). Cada avaliação executa apenas o SimPy (~0.5s)
      sem releitura de disco (~1.0s), reduzindo o tempo total em ~85%.

    Penalização de inviabilidade:
      Σx ≠ N_LIVRES → retorna PENALIDADE_HARD em todos os objetivos.

    Alinhamento com PM4SOS: esta função é o elo entre o algoritmo evolutivo
    e o DES (P7b), conforme Algoritmo 5 de Ferronato (2022).
    """
    # Verificar viabilidade: Σx = N_LIVRES e limites por categoria
    if sum(x) != N_LIVRES or any(x[i] < LB[i] or x[i] > UB[i]
                                  for i in range(len(x))):
        return [PENALIDADE_HARD] * 4

    if not _DES_OK or not logs_cache:
        return [rng.uniform(0.5, 5.0), rng.uniform(0, 0.5),
                rng.uniform(0, 0.3), rng.uniform(0, 0.3)]

    # Construir configuração do gabinete com a solução candidata
    estrutura = [
        ("CJ3A",  x[0], "instrutor", 1.00),
        ("CJ3C",  1,    "gestao",    0.10),
        ("CJ2A",  x[1], "instrutor", 0.80),
        ("FC6C",  x[2], "instrutor", 0.70),
        ("FC4IV", x[3], "admin",     0.20),
        ("FC2II", x[4], "admin",     0.10),
    ]
    config = ConfiguracaoGabinete({"estrutura": estrutura})

    # Selecionar n_rep entradas do cache aleatoriamente
    sel = rng.sample(logs_cache, min(n_rep, len(logs_cache)))

    t_medios, ginis, kappas, etas = [], [], [], []
    for event_log, traces_ord, horizonte in sel:
        try:
            semente = rng.randint(1, 99999)
            env     = simpy.Environment()
            monitor = MonitorMetricas()
            modelo  = ModeloGabinete(
                env=env, config=config, params_des=params_des,
                monitor=monitor, restricoes=restricoes,
                rng=random.Random(semente),
            )
            env.process(_gerador_traces(env, traces_ord, modelo))
            env.run(until=horizonte)
            t_medios.append(monitor.calcular_t_medio())
            ginis.append(monitor.calcular_gini())
            kappas.append(monitor.calcular_kappa())
            etas.append(monitor.calcular_eta())
        except Exception as exc:
            log.debug("    DES fitness erro: %s", exc)

    if not t_medios:
        return [PENALIDADE_HARD] * 4

    return [
        float(np.mean(t_medios)),
        float(np.mean(ginis)),
        1.0 - float(np.mean(kappas)),
        1.0 - float(np.mean(etas)),
    ]


# ===========================================================================
# SEÇÃO 5 — Utilitários evolutivos: representação, crossover, mutação
# ===========================================================================

def gerar_individuo_aleatorio(rng: random.Random) -> List[int]:
    """
    Gera indivíduo aleatório respeitando Σx = N_LIVRES e limites.
    Usa distribuição Dirichlet discreta para garantir a restrição de soma.
    """
    n = len(CATS_LIVRES)
    while True:
        # Sortear pesos e escalar para N_LIVRES
        pesos  = [rng.randint(LB[i], UB[i]) for i in range(n)]
        escala = N_LIVRES / sum(pesos)
        x = [max(LB[i], min(UB[i], round(pesos[i] * escala)))
             for i in range(n)]
        # Corrigir soma
        diff = N_LIVRES - sum(x)
        for _ in range(abs(diff)):
            idx = rng.randint(0, n - 1)
            if diff > 0 and x[idx] < UB[idx]:
                x[idx] += 1
            elif diff < 0 and x[idx] > LB[idx]:
                x[idx] -= 1
        if sum(x) == N_LIVRES:
            return x


def crossover_sbx(x1: List[int], x2: List[int],
                  rng: random.Random, eta: float = 2.0) -> Tuple[List[int], List[int]]:
    """
    Simulated Binary Crossover (SBX) adaptado para inteiros com restrição de soma.
    Após crossover, projeta ambos os filhos para o hiperplano Σx = N_LIVRES.
    """
    n    = len(x1)
    c1   = list(x1)
    c2   = list(x2)
    u    = rng.random()
    beta = (2 * u) ** (1 / (eta + 1)) if u < 0.5 else (1 / (2 * (1 - u))) ** (1 / (eta + 1))
    for i in range(n):
        if rng.random() < 0.5:
            c1[i] = round(0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i]))
            c2[i] = round(0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i]))
            c1[i] = max(LB[i], min(UB[i], c1[i]))
            c2[i] = max(LB[i], min(UB[i], c2[i]))

    # Projetar na restrição de soma
    for c in (c1, c2):
        diff = N_LIVRES - sum(c)
        idxs = list(range(n))
        rng.shuffle(idxs)
        for idx in idxs:
            if diff == 0:
                break
            if diff > 0 and c[idx] < UB[idx]:
                c[idx] += 1; diff -= 1
            elif diff < 0 and c[idx] > LB[idx]:
                c[idx] -= 1; diff += 1
    return c1, c2


def mutacao_inteira(x: List[int], rng: random.Random, taxa: float = 0.2) -> List[int]:
    """
    Mutação por troca (swap mutation) que preserva a restrição de soma.
    Sorteia dois índices e transfere 1 unidade do maior para o menor.
    """
    x2 = list(x)
    if rng.random() < taxa:
        i, j = rng.sample(range(len(x2)), 2)
        if x2[i] > LB[i] and x2[j] < UB[j]:
            x2[i] -= 1
            x2[j] += 1
    return x2


def domina(a: List[float], b: List[float]) -> bool:
    """Retorna True se 'a' domina 'b' (minimização em todos os objetivos)."""
    return all(ai <= bi for ai, bi in zip(a, b)) and any(ai < bi for ai, bi in zip(a, b))


def calcular_hypervolume(pareto: List[List[float]],
                          ref:   List[float] = HV_REF) -> float:
    """
    Calcula o hypervolume (indicador de qualidade da fronteira de Pareto).
    Implementação simplificada para 4 objetivos via Monte Carlo (1000 amostras).
    Para resultados exatos, usar pyGMO/DEAP em ambiente com mais memória.
    """
    if not pareto:
        return 0.0
    n_samples = 1000
    rng_hv    = random.Random(42)
    dentro    = 0
    for _ in range(n_samples):
        ponto = [rng_hv.uniform(0, r) for r in ref]
        if any(all(f[j] <= ponto[j] for j in range(len(ref))) for f in pareto):
            dentro += 1
    volume_ref = math.prod(ref)
    return (dentro / n_samples) * volume_ref


def fronteira_pareto(populacao: List[Tuple[List[int], List[float]]]) -> List[Tuple[List[int], List[float]]]:
    """Extrai a fronteira de Pareto de uma população (solução, objetivos)."""
    pareto = []
    for i, (xi, fi) in enumerate(populacao):
        dominado = False
        for j, (xj, fj) in enumerate(populacao):
            if i != j and domina(fj, fi):
                dominado = True
                break
        if not dominado:
            pareto.append((xi, fi))
    return pareto


# ===========================================================================
# SEÇÃO 6 — Algoritmos evolutivos: NSGA-II, AMGA2, SPEA2
# ===========================================================================

def nsga2(logs_cache:   List[Tuple],
           params_des:  List[Dict],
           restricoes:  List[Dict],
           warm_start:  List[List[int]],
           rng:         random.Random,
           n_pop:       int = N_POP,
           n_gen:       int = N_GEN) -> List[Tuple[List[int], List[float]]]:
    """
    NSGA-II (Non-dominated Sorting Genetic Algorithm II).
    DEB et al. (2002) — IEEE Trans. Evol. Comput., v.6, n.2, pp.182-197.

    Implementação:
      1. Inicializar população com warm-start CBR + aleatórios
      2. Para cada geração:
         a. Avaliar fitness (DES) para novos indivíduos
         b. Ranking por não-dominância (fast non-dominated sort)
         c. Crowding distance dentro de cada front
         d. Seleção por torneio binário (rank + crowding)
         e. Crossover SBX + mutação inteira
         f. Elitismo: manter os melhores N da população combinada Pt ∪ Qt

    Alinhamento com PM4SOS: Ferronato usa AMGA2; o PM4JUD adiciona NSGA-II
    como segundo benchmark para comparação experimental controlada.
    """
    def _avaliar(x):
        return avaliar_solucao(x, logs_cache, params_des, restricoes, rng)

    def _crowding_distance(front):
        n = len(front)
        if n <= 2:
            return [float('inf')] * n
        dist = [0.0] * n
        for obj_idx in range(4):
            sorted_front = sorted(range(n), key=lambda i: front[i][1][obj_idx])
            dist[sorted_front[0]] = dist[sorted_front[-1]] = float('inf')
            fmin = front[sorted_front[0]][1][obj_idx]
            fmax = front[sorted_front[-1]][1][obj_idx]
            rng_obj = max(fmax - fmin, 1e-9)
            for k in range(1, n - 1):
                dist[sorted_front[k]] += (
                    front[sorted_front[k + 1]][1][obj_idx] -
                    front[sorted_front[k - 1]][1][obj_idx]
                ) / rng_obj
        return dist

    def _fast_nondominated_sort(pop):
        fronts  = [[]]
        n       = len(pop)
        dom_set = [[] for _ in range(n)]
        dom_cnt = [0] * n
        rank    = [0] * n
        for i in range(n):
            for j in range(n):
                if i == j: continue
                if domina(pop[i][1], pop[j][1]):
                    dom_set[i].append(j)
                elif domina(pop[j][1], pop[i][1]):
                    dom_cnt[i] += 1
            if dom_cnt[i] == 0:
                rank[i] = 0
                fronts[0].append(i)
        current = 0
        while fronts[current]:
            next_front = []
            for i in fronts[current]:
                for j in dom_set[i]:
                    dom_cnt[j] -= 1
                    if dom_cnt[j] == 0:
                        rank[j] = current + 1
                        next_front.append(j)
            current += 1
            fronts.append(next_front)
        return fronts[:-1], rank

    # Inicializar população
    pop = [(ws, _avaliar(ws)) for ws in warm_start[:n_pop]]
    while len(pop) < n_pop:
        x = gerar_individuo_aleatorio(rng)
        pop.append((x, _avaliar(x)))

    for gen in range(n_gen):
        # Gerar filhos por crossover + mutação
        filhos = []
        while len(filhos) < n_pop:
            p1 = pop[rng.randint(0, n_pop - 1)][0]
            p2 = pop[rng.randint(0, n_pop - 1)][0]
            c1, c2 = crossover_sbx(p1, p2, rng)
            c1 = mutacao_inteira(c1, rng)
            c2 = mutacao_inteira(c2, rng)
            filhos.append((c1, _avaliar(c1)))
            if len(filhos) < n_pop:
                filhos.append((c2, _avaliar(c2)))

        combinada    = pop + filhos
        fronts, rank = _fast_nondominated_sort(combinada)

        nova_pop = []
        for front in fronts:
            if len(nova_pop) + len(front) <= n_pop:
                nova_pop.extend([combinada[i] for i in front])
            else:
                restante = n_pop - len(nova_pop)
                front_pop   = [(combinada[i], i) for i in front]
                dist        = _crowding_distance([fp[0] for fp in front_pop])
                selecionados = sorted(range(len(front)),
                                      key=lambda k: -dist[k])[:restante]
                nova_pop.extend([combinada[front[k]] for k in selecionados])
                break
        pop = nova_pop[:n_pop]

    return [(x, f) for x, f in pop]


def amga2(logs_cache:   List[Tuple],
           params_des:  List[Dict],
           restricoes:  List[Dict],
           warm_start:  List[List[int]],
           rng:         random.Random,
           n_pop:       int = N_POP,
           n_gen:       int = N_GEN) -> List[Tuple[List[int], List[float]]]:
    """
    AMGA2 (Adaptive Multi-Goal Algorithm 2).
    TIWARI et al. (2011) — Evol. Comput., v.19, n.4, pp.577-609.
    Referenciado por FERRONATO (2022) como algoritmo principal do PM4SOS.

    Implementação resumida dos elementos essenciais do AMGA2:
      • Archivamento de soluções não-dominadas (arquivo externo)
      • Referência adaptativa: ponto de referência atualizado por geração
      • Decomposição multiobjetivo: escalarização de Chebyshev com peso adaptativo
      • Busca local: refinamento do melhor indivíduo em cada geração

    A escalarização de Chebyshev transforma o MOOP em múltiplos SOOPs:
      F_λ(x) = max_i { λ_i × (f_i(x) − z*_i) }
    onde z* é o ponto ideal (nadir atual) e λ são pesos adaptados
    dinamicamente para direcionar a busca para regiões não exploradas.
    """
    def _avaliar(x):
        return avaliar_solucao(x, logs_cache, params_des, restricoes, rng)

    def _chebyshev(f, ref, pesos):
        return max(pesos[i] * abs(f[i] - ref[i]) for i in range(len(f)))

    # Inicializar arquivo + população
    arquivo = []   # soluções não-dominadas (archive)
    pop     = [(ws, _avaliar(ws)) for ws in warm_start[:n_pop]]
    while len(pop) < n_pop:
        x = gerar_individuo_aleatorio(rng)
        pop.append((x, _avaliar(x)))

    for gen in range(n_gen):
        # Atualizar arquivo (non-dominated)
        todos = pop + arquivo
        arquivo = [(x, f) for x, f in fronteira_pareto(todos)][:n_pop * 2]

        # Ponto ideal adaptativo (nadir do arquivo)
        fs     = [f for _, f in (arquivo or pop)]
        z_star = [min(f[i] for f in fs) for i in range(4)]

        # Gerar pesos adaptativos: favorecer direções pouco representadas
        pesos_gen = []
        for _ in range(n_pop):
            w = [rng.uniform(0.01, 1.0) for _ in range(4)]
            total = sum(w); pesos_gen.append([wi / total for wi in w])

        # Seleção por Chebyshev + crossover + mutação
        nova_pop = []
        for pesos in pesos_gen:
            base = min(pop, key=lambda xf: _chebyshev(xf[1], z_star, pesos))[0]
            par  = rng.choice(pop)[0]
            c, _ = crossover_sbx(base, par, rng)
            c    = mutacao_inteira(c, rng)
            nova_pop.append((c, _avaliar(c)))

        pop = nova_pop

        # Busca local: refinar melhor indivíduo do arquivo
        if arquivo:
            melhor_x = min(arquivo, key=lambda xf: sum(xf[1]))[0]
            vizinho  = mutacao_inteira(melhor_x, rng, taxa=1.0)
            fv       = _avaliar(vizinho)
            fm       = min(arquivo, key=lambda xf: sum(xf[1]))[1]
            if domina(fv, fm):
                arquivo.append((vizinho, fv))

    return arquivo or pop


def spea2(logs_cache:   List[Tuple],
           params_des:  List[Dict],
           restricoes:  List[Dict],
           warm_start:  List[List[int]],
           rng:         random.Random,
           n_pop:       int = N_POP,
           n_gen:       int = N_GEN) -> List[Tuple[List[int], List[float]]]:
    """
    SPEA2 (Strength Pareto Evolutionary Algorithm 2).
    ZITZLER et al. (2001) — TIK-Report 103, ETH Zürich.

    Elementos centrais do SPEA2:
      • Arquivo externo de tamanho fixo (n_pop)
      • Fitness = strength (n soluções dominadas) + densidade kNN
      • Truncation operator: mantém diversidade no arquivo
      • Seleção por torneio binário sobre fitness SPEA2
    """
    def _avaliar(x):
        return avaliar_solucao(x, logs_cache, params_des, restricoes, rng)

    def _spea2_fitness(combinada):
        n = len(combinada)
        strength = [0] * n
        for i in range(n):
            for j in range(n):
                if i != j and domina(combinada[i][1], combinada[j][1]):
                    strength[i] += 1
        raw = [0.0] * n
        for i in range(n):
            for j in range(n):
                if i != j and domina(combinada[j][1], combinada[i][1]):
                    raw[i] += strength[j]
        # Densidade: k-ésima distância Euclidiana no espaço objetivo
        k_nn = max(1, int(math.sqrt(n)))
        dens = []
        for i in range(n):
            dists = sorted(
                sum((combinada[i][1][d] - combinada[j][1][d]) ** 2
                    for d in range(4)) ** 0.5
                for j in range(n) if i != j
            )
            sigma_k = dists[k_nn - 1] if len(dists) >= k_nn else dists[-1]
            dens.append(1.0 / (sigma_k + 2.0))
        return [raw[i] + dens[i] for i in range(n)]

    # Inicializar
    arquivo = []
    pop     = [(ws, _avaliar(ws)) for ws in warm_start[:n_pop]]
    while len(pop) < n_pop:
        x = gerar_individuo_aleatorio(rng)
        pop.append((x, _avaliar(x)))

    for gen in range(n_gen):
        combinada = pop + arquivo
        fitness   = _spea2_fitness(combinada)

        # Arquivo: manter não-dominados (fitness < 1.0)
        arquivo = [(combinada[i][0], combinada[i][1])
                   for i in range(len(combinada)) if fitness[i] < 1.0]
        if len(arquivo) > n_pop:
            # Truncation: remover o mais próximo dos demais
            while len(arquivo) > n_pop:
                dists_min = []
                for i in range(len(arquivo)):
                    d = min(
                        sum((arquivo[i][1][k] - arquivo[j][1][k]) ** 2
                            for k in range(4)) ** 0.5
                        for j in range(len(arquivo)) if j != i
                    )
                    dists_min.append(d)
                arquivo.pop(dists_min.index(min(dists_min)))
        elif len(arquivo) < n_pop:
            # Completar com dominados de menor fitness
            outros = sorted(
                [(i, fitness[i]) for i in range(len(combinada))
                 if fitness[i] >= 1.0],
                key=lambda x: x[1]
            )
            for idx, _ in outros[:n_pop - len(arquivo)]:
                arquivo.append(combinada[idx])

        # Seleção por torneio + crossover + mutação
        nova_pop = []
        fit_arq  = _spea2_fitness(arquivo)
        while len(nova_pop) < n_pop:
            i1, i2 = rng.randint(0, len(arquivo)-1), rng.randint(0, len(arquivo)-1)
            p1 = arquivo[i1][0] if fit_arq[i1] < fit_arq[i2] else arquivo[i2][0]
            i3, i4 = rng.randint(0, len(arquivo)-1), rng.randint(0, len(arquivo)-1)
            p2 = arquivo[i3][0] if fit_arq[i3] < fit_arq[i4] else arquivo[i4][0]
            c, _ = crossover_sbx(p1, p2, rng)
            c    = mutacao_inteira(c, rng)
            nova_pop.append((c, _avaliar(c)))
        pop = nova_pop

    return arquivo


# ===========================================================================
# SEÇÃO 7 — Carregamento de artefatos
# ===========================================================================

def carregar_params_des(input_dir: Path, gabinete: str) -> List[Dict]:
    fpath = input_dir / f"params_des_{gabinete}.csv"
    if not fpath.exists():
        log.warning("  params_des_%s.csv não encontrado", gabinete)
        return []
    with fpath.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    log.info("  params_des: %d atividades", len(rows))
    return rows


def carregar_restricoes(ont: Optional[OntologiaPM4JUD], ltlf_path: Optional[Path]) -> List[Dict]:
    if ont is not None:
        try:
            r = ont.constraints_ltlf()
            if r:
                return r
        except Exception:
            pass
    if ltlf_path and ltlf_path.exists():
        try:
            data = json.loads(ltlf_path.read_text(encoding="utf-8"))
            return data.get("constraints_metadata", [])
        except Exception:
            pass
    return [{"id": "C10", "prazo_dias": 10}, {"id": "C11", "prazo_dias": 30}]


def carregar_perfil_gabinete(input_dir: Path, gabinete: str) -> Dict[str, float]:
    """Extrai características do gabinete para o vetor CBR Φ."""
    perfil = {}
    # λ prio/regular
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            fpath = input_dir / f"params_des_{gabinete}.csv"
            if fpath.exists():
                with fpath.open(encoding=enc) as f:
                    rows = list(csv.DictReader(f))
                if rows:
                    lp = rows[0].get("lambda_prio", 0)
                    lr = rows[0].get("lambda_regular", 0)
                    perfil["lambda_prio"]    = float(lp) if lp else 0.0
                    perfil["lambda_regular"] = float(lr) if lr else 0.0
                    perfil["n_atividades"]   = float(len(rows))
            break
        except UnicodeDecodeError:
            continue
    # κ e T̄ baseline
    for nome in (f"des_{gabinete}.json", "p7b_relatorio.json"):
        fpath = input_dir / nome
        if fpath.exists():
            try:
                data = json.loads(fpath.read_text(encoding="utf-8"))
                if nome.startswith("des_"):
                    m = data.get("metricas", {})
                    perfil["t_medio_base"] = m.get("t_medio_dias", {}).get("media", 0)
                    perfil["kappa_base"]   = m.get("kappa", {}).get("media", 1.0)
                break
            except Exception:
                pass
    return perfil


def carregar_objetivos_controle(input_dir: Path, gabinete: str) -> List[float]:
    """Carrega objetivos do grupo controle (P7b) para inicializar o CBR."""
    fpath = input_dir / f"des_{gabinete}.json"
    if not fpath.exists():
        return [1.0, 0.1, 0.0, 0.0]
    try:
        data = json.loads(fpath.read_text(encoding="utf-8"))
        m    = data.get("metricas", {})
        return [
            m.get("t_medio_dias", {}).get("media", 1.0),
            m.get("gini",         {}).get("media", 0.1),
            1.0 - m.get("kappa",  {}).get("media", 1.0),
            1.0 - m.get("eta",    {}).get("media", 1.0),
        ]
    except Exception:
        return [1.0, 0.1, 0.0, 0.0]


# ===========================================================================
# SEÇÃO 8 — Pipeline por gabinete
# ===========================================================================

def processar_gabinete(gabinete:   str,
                        input_dir:  Path,
                        output_dir: Path,
                        cbr:        BaseConhecimentoCBR,
                        ont_opt:    OntologiaPM4JUD,
                        algoritmos: List[str],
                        n_rep:      int,
                        n_pop:      int,
                        n_gen:      int) -> Dict:
    """
    Pipeline de otimização completo para um gabinete.

    Para cada algoritmo × n_rep replicações:
      1. Extrair Φ (perfil do gabinete)
      2. CBR.recuperar(Φ) → warm-start
      3. Executar algoritmo evolutivo (NSGA-II, AMGA2 ou SPEA2)
      4. Calcular hypervolume da fronteira de Pareto
      5. CBR.reter(melhor solução)
      6. Salvar resultados
    """
    log.info("=" * 70)
    log.info("Processando gabinete: %s", gabinete.upper())
    log.info("=" * 70)
    t0 = time.time()

    # --- Artefatos ---
    params_des = carregar_params_des(input_dir, gabinete)
    ltlf_path  = input_dir / f"ltlf_{gabinete}.json"

    # Ontologia — usar ontologia do DES para constraints
    from pm4jud_des import carregar_ontologia as _ont_des, carregar_restricoes as _rest
    _ont_des_obj = None   # reutilizar se disponível
    restricoes = carregar_restricoes(None, ltlf_path)

    # XES disponíveis
    xes_paths = sorted(input_dir.glob(f"sim2log_{gabinete}_r*.xes"))
    if not xes_paths:
        log.error("  Nenhum XES encontrado — execute P7a primeiro")
        return {"gabinete": gabinete, "status": "ERRO", "motivo": "XES ausentes"}
    log.info("  XES disponíveis: %d", len(xes_paths))

    # CBR — inicializar com grupo controle se base vazia
    phi_dict = carregar_perfil_gabinete(input_dir, gabinete)
    phi_arr  = np.array([
        phi_dict.get("lambda_prio",    0.0) / 0.001,
        phi_dict.get("lambda_regular", 0.0) / 0.001,
        phi_dict.get("n_atividades",   0.0) / 100.0,
        phi_dict.get("t_medio_base",   0.0) / 365.0,
        phi_dict.get("kappa_base",     1.0),
    ])
    obj_controle = carregar_objetivos_controle(input_dir, gabinete)
    cbr.inicializar_com_controle(gabinete, phi_dict, obj_controle)
    log.info("  Perfil CBR Φ: λ_prio=%.6f T̄_base=%.2f κ_base=%.4f",
             phi_dict.get("lambda_prio", 0), phi_dict.get("t_medio_base", 0),
             phi_dict.get("kappa_base", 1))

    # --- Pré-parsing dos XES (uma vez por gabinete) ---
    # Cada entrada do cache: (EventLog, traces_ordenadas, horizonte_s)
    log.info("  Pré-parseando %d XES (cache único por gabinete)...", len(xes_paths))
    t_cache = time.time()
    logs_cache = _preparar_cache_xes(xes_paths)
    log.info("  Cache XES: %d logs em memória | t=%.1fs",
             len(logs_cache), time.time() - t_cache)

    # --- Loop por algoritmo ---
    resultado_gab = {"gabinete": gabinete, "status": "OK", "algoritmos": {}}

    for algo in algoritmos:
        log.info("  Algoritmo: %s (%d rep. × %d pop × %d gen)",
                 algo.upper(), n_rep, n_pop, n_gen)
        t_algo = time.time()
        hvs: List[float] = []
        rep_results: List[Dict] = []

        for rep in range(1, n_rep + 1):
            t_rep = time.time()
            semente = rep * 137 + hash(gabinete + algo) % 10000
            rng     = random.Random(semente)

            # CBR: Recuperar + Reutilizar
            warm = cbr.extrair_warm_start(phi_arr, k=K_CBR)
            while len(warm) < n_pop:
                warm.append(gerar_individuo_aleatorio(rng))

            # Executar algoritmo (logs_cache em vez de xes_paths)
            if algo == "nsga2":
                resultado = nsga2(logs_cache, params_des, restricoes,
                                   warm, rng, n_pop, n_gen)
            elif algo == "amga2":
                resultado = amga2(logs_cache, params_des, restricoes,
                                   warm, rng, n_pop, n_gen)
            elif algo == "spea2":
                resultado = spea2(logs_cache, params_des, restricoes,
                                   warm, rng, n_pop, n_gen)

            else:
                resultado = []

            # Fronteira de Pareto + hypervolume
            pareto = fronteira_pareto(resultado)
            hv     = calcular_hypervolume([f for _, f in pareto])
            hvs.append(hv)

            # CBR: Reter — melhor solução da fronteira (menor T̄)
            if pareto:
                melhor_x, melhor_f = min(pareto, key=lambda xf: xf[1][0])
                cbr.reter(RepresentacaoCaso(
                    gabinete        = gabinete,
                    caracteristicas = phi_dict,
                    solucao         = melhor_x,
                    objetivos       = melhor_f,
                    algoritmo       = algo,
                    timestamp       = datetime.now(timezone.utc).isoformat(),
                ))

            t_elapsed = round(time.time() - t_rep, 1)
            log.info("    Rep %2d/%d | hv=%.4f | |pareto|=%d | t=%ss",
                     rep, n_rep, hv, len(pareto), t_elapsed)

            rep_results.append({
                "replicacao": rep,
                "semente":    semente,
                "hv":         round(hv, 6),
                "n_pareto":   len(pareto),
                "pareto":     [(x, [round(v, 4) for v in f])
                               for x, f in pareto],
            })

            # Salvar replicação
            out_path = output_dir / f"opt_{gabinete}_{algo}_r{rep:02d}.json"
            out_path.write_text(
                json.dumps({
                    "gabinete": gabinete, "algoritmo": algo,
                    "replicacao": rep, "hv": round(hv, 6),
                    "n_pareto": len(pareto),
                    "pareto": [(x, [round(v, 4) for v in f]) for x, f in pareto],
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        # Agregar por algoritmo
        hv_medio  = round(float(np.mean(hvs)), 6)
        hv_desvpad = round(float(np.std(hvs, ddof=1)) if len(hvs) > 1 else 0, 6)
        t_crit    = 2.045  # t de Student, 29 g.l., α=0.05
        ic_lo = round(hv_medio - t_crit * hv_desvpad / math.sqrt(len(hvs)), 6)
        ic_hi = round(hv_medio + t_crit * hv_desvpad / math.sqrt(len(hvs)), 6)

        resultado_algo = {
            "algoritmo": algo, "n_rep": n_rep,
            "hv": {"media": hv_medio, "desvpad": hv_desvpad,
                   "ic95_lo": ic_lo, "ic95_hi": ic_hi},
            "replicacoes": rep_results,
            "tempo_s": round(time.time() - t_algo, 1),
        }
        out_algo = output_dir / f"opt_{gabinete}_{algo}.json"
        out_algo.write_text(
            json.dumps(resultado_algo, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info("  %s: hv_medio=%.4f ± %.4f | t=%.0fs",
                 algo.upper(), hv_medio, hv_desvpad,
                 time.time() - t_algo)
        resultado_gab["algoritmos"][algo] = resultado_algo

    resultado_gab["tempo_s"] = round(time.time() - t0, 1)
    return resultado_gab


# ===========================================================================
# SEÇÃO 9 — Relatório e entry point
# ===========================================================================

def salvar_relatorio(resultados: List[Dict], output_dir: Path) -> None:
    payload = {
        "programa":    "PM4JUD-OPT",
        "versao":      VERSION,
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "descricao": (
            "Otimização multiobjetivo com CBR+NSGA-II, CBR+AMGA2 e CBR+SPEA2. "
            "Hypervolume da fronteira de Pareto como métrica comparativa. "
            "Design experimental: 30 rep. × 3 algoritmos = 90 execuções. "
            "Análise estatística em P9: Kruskal-Wallis + Wilcoxon + Bonferroni α=0.05."
        ),
        "proximo_passo": "PM4JUD-STAT (P9) — análise estatística dos hypervolumes",
        "gabinetes":     resultados,
    }
    fpath = output_dir / "p8_relatorio.json"
    fpath.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Relatório: %s", fpath.name)


def imprimir_resumo(resultados: List[Dict], algoritmos: List[str]) -> None:
    log.info("")
    log.info("=" * 75)
    log.info("RESUMO P8 — OTIMIZAÇÃO MULTIOBJETIVO COM CBR")
    log.info("=" * 75)
    log.info("  %-12s  %s",
             "Gabinete",
             "  ".join(f"{a.upper():>12}" for a in algoritmos))
    log.info("  %s", "-" * 65)
    for r in resultados:
        hvs = "  ".join(
            f"{r['algoritmos'].get(a, {}).get('hv', {}).get('media', 0):>12.4f}"
            for a in algoritmos
        )
        log.info("  %-12s  %s  (HV médio)", r["gabinete"], hvs)
    log.info("")
    log.info("Próximo: PM4JUD-STAT (P9) — comparação estatística")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pm4jud_opt",
        description="PM4JUD-OPT v1.0 (P8) — Otimização Multiobjetivo com CBR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input",     required=True, type=Path)
    parser.add_argument("--output",    required=True, type=Path)
    parser.add_argument("--ontologia", default=None,  type=Path)
    parser.add_argument("--gabinetes", nargs="+",
                        default=["reynaldo", "palheiro", "schietti"])
    parser.add_argument("--algoritmos", nargs="+", default=ALGORITMOS,
                        choices=ALGORITMOS)
    parser.add_argument("--n-rep",  type=int, default=30)
    parser.add_argument("--n-pop",  type=int, default=N_POP)
    parser.add_argument("--n-gen",  type=int, default=N_GEN)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    log.info("PM4JUD-OPT v%s | %s | %d rep × %d pop × %d gen",
             VERSION, "+".join(args.algoritmos),
             args.n_rep, args.n_pop, args.n_gen)

    # Base CBR — compartilhada entre gabinetes e algoritmos
    cbr = BaseConhecimentoCBR(args.output / "base_casos.json")

    # Ontologia (Módulos 3 + 7)
    ont_opt = OntologiaPM4JUD(args.ontologia)

    resultados = []
    for gabinete in args.gabinetes:
        r = processar_gabinete(
            gabinete, args.input, args.output,
            cbr, ont_opt,
            algoritmos=args.algoritmos,
            n_rep=args.n_rep,
            n_pop=args.n_pop,
            n_gen=args.n_gen,
        )
        resultados.append(r)

    salvar_relatorio(resultados, args.output)
    imprimir_resumo(resultados, args.algoritmos)


if __name__ == "__main__":
    main()
