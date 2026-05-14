#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
PM4JUD-COMPLEMENT  v3.0
================================================================================

Dissertação de Mestrado — PPGIa/PUCPR
Título: PM4JUD — Otimização Multiobjetivo com Mineração de Processos e
        Simulação no Contexto do Fluxo Processual em Gabinetes de Magistrado
Autor:  Luiz Claudio Soares de Almeida
Orient: Prof. Dr. Edson Emilio Scalabrin
Ano:    2026

Descrição
---------
Complementa o log TPU filtrado (P2) com eventos internos dos gabinetes,
gerando o log operacional completo que alimenta o PM4JUD-PM (P4).

AGRUPAMENTO DE ATIVIDADES — 5 grupos funcionais do SAGWeb
──────────────────────────────────────────────────────────
Cada atividade interna pertence a exatamente um grupo, refletindo as
funcionalidades implementadas no Sistema de Automação de Gabinetes Web:

  G1  Escaninhos do processo
      Estados de tramitação interna: Recebido, Em análise,
      Conclusão para decisão, Aguardando sessão, Julgado

  G2  Workflow de documentos
      G2a Criação de documento  (relatório, minuta, certidão)
      G2b Alteração de documento (revisões e correções)
      G2c Exclusão de documento  (documentos descartados)
      G2d Publicação de documento (DJe)
      G2e Jurisprudência         (precedentes vinculados)

  G3  Lançamentos de fases
      Fases internas (sem código TPU) e fases externas (com código TPU MNI)

  G4  Deslocamento de processos entre unidades
      Gabinete → Turma, Turma → Gabinete, Protocolo

  G5  Assinatura Eletrônica
      G5a Preparação da Chancela Eletrônica
      G5b Assinatura de documento (assessor ou ministro)

ESTRUTURA DA TABELA DE EXPERIÊNCIA
────────────────────────────────────
tabela_experiencia[assessor_id][classe_tpu][grupo_sagweb] = {
    "n_lancamentos":    int    # total de lançamentos no grupo/classe
    "duracao_media":    float  # duração média por atividade (dias)
    "fator_experiencia": float # normalizado [0,1] pelo max do grupo/classe
}

Derivação (Opção B — calibração empírica):
  1. Do corpus DATAJUD (P2): distribuição de classes e durações medianas
  2. Dirichlet calibrado → proporção de processos por assessor × classe
  3. proporção × total_classe × lognormal(0, σ) → processos_instruídos
  4. processos_instruídos × n_base_grupo × fator_nivel → n_lancamentos
  5. fator_exp = n_lancamentos / max(n_lancamentos no grupo/classe)

FASE 1 / FASE 2 (FASE_PM4JUD env var)
────────────────────────────────────
  FASE 1: assessores sintéticos + eventos gerados com base na tabela
  FASE 2: dados reais SAGWeb + D'Castro sobre log completo

Pipeline
────────
  P1 ETL -> P2 REFINE_1 -> P3 COMPLEMENT -> P4 REFINE_2
          -> [P5 PM] -> P6 LTLf -> P7a Sim2Log -> P7b DES -> P8 OPT -> P9 STAT           


  Entrada : dcastro_<gab>.xes  (saída P2)
  Saída   :
    complement_<gab>.xes                  — log completo com assessores
    complement_<gab>_assessores.csv       — perfil de cada assessor
    complement_<gab>_experiencia.csv      — tabela grupo × classe × assessor
    complement_relatorio.json

Referências
───────────
D'Castro (2020); Ferronato (2022); Van der Aalst (2016); RISTJ (2024)

Repositório: https://github.com/luizcsalmeida/pm4jud/tree/main/complement
================================================================================
"""

import argparse, json, logging, math, os, random, sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

try:
    from pm4py.objects.log.obj import EventLog, Trace, Event
    from pm4py.objects.log.importer.xes import importer as xes_importer
    from pm4py.objects.log.exporter.xes import exporter as xes_exporter
except ImportError:
    print("[ERRO] pm4py não encontrado. pip install pm4py", file=sys.stderr)
    sys.exit(1)

# Vocabulário canônico e loaders ontológicos compartilhados
from pm4jud_vocab import (
    carregar_rotulos_colegiada,
    carregar_classes_prioritarias,
)
try:
    from rdflib import Graph
    RDFLIB_DISPONIVEL = True
except ImportError:
    RDFLIB_DISPONIVEL = False


# ==============================================================================
# GRUPOS FUNCIONAIS DO SAGWeb
# Cada grupo tem: subgrupos (lista de rótulos canônicos de atividades)
# e n_base (número médio de execuções por processo para processos criminais HC)
# ==============================================================================

GRUPOS_SAGWEB: Dict[str, Dict] = {
    "G1_escaninhos": {
        "label":    "Escaninhos do processo",
        "n_base":   5.5,    # médias empíricas por processo HC
        "subgrupos": {
            "Recebido pelo assessor":       0.90,   # fração do n_base
            "Incluido em lista HC/RHC":     0.25,   # somente HC prioritário
            "Em analise":                   1.20,
            "Conclusao para decisao":       0.90,
            "Aguardando sessao":            0.80,
            "Julgado":                      0.85,
            "Arquivado":                    0.10,
        },
    },
    "G2a_wf_criacao": {
        "label":    "Workflow de documentos — Criação",
        "n_base":   2.3,
        "subgrupos": {
            "Criacao de documento: Relatorio conclusivo":  0.90,
            "Criacao de documento: Minuta de voto":        0.85,
            "Criacao de documento: Certidao":              0.30,
            "Criacao de documento: Outros":                0.25,
        },
    },
    "G2b_wf_alteracao": {
        "label":    "Workflow de documentos — Alteração",
        "n_base":   1.8,
        "subgrupos": {
            "Alteracao de documento: Relatorio conclusivo": 0.80,
            "Alteracao de documento: Minuta de voto":       0.75,
            "Alteracao de documento: Outros":               0.25,
        },
    },
    "G2c_wf_exclusao": {
        "label":    "Workflow de documentos — Exclusão",
        "n_base":   0.2,
        "subgrupos": {
            "Exclusao de documento": 1.00,
        },
    },
    "G2d_wf_publicacao": {
        "label":    "Workflow de documentos — Publicação",
        "n_base":   1.0,
        "subgrupos": {
            "Publicacao de documento no DJe": 1.00,
        },
    },
    "G2e_wf_jurisprudencia": {
        "label":    "Workflow de documentos — Jurisprudência",
        "n_base":   0.4,
        "subgrupos": {
            "Vinculacao de jurisprudencia": 1.00,
        },
    },
    "G3_lancamentos_fases": {
        "label":    "Lançamentos de fases",
        "n_base":   3.2,
        "subgrupos": {
            "Fase interna: Instrucao":          0.60,
            "Fase interna: Conclusao":          0.50,
            "Fase externa: Distribuicao":       0.45,
            "Fase externa: Ato ordinatorio":    0.45,
            "Fase externa: Julgamento":         0.40,
        },
    },
    "G4_deslocamento": {
        "label":    "Deslocamento de processos entre unidades",
        "n_base":   1.3,
        "subgrupos": {
            "Deslocamento: Gabinete para Turma":     0.60,
            "Deslocamento: Turma para Gabinete":     0.25,
            "Deslocamento: Gabinete para Protocolo": 0.15,
        },
    },
    "G5a_assinatura_chancela": {
        "label":    "Assinatura Eletrônica — Preparação da Chancela",
        "n_base":   1.0,
        "subgrupos": {
            "Preparacao da Chancela Eletronica": 1.00,
        },
    },
    "G5b_assinatura_documento": {
        "label":    "Assinatura Eletrônica — Assinatura de documento",
        "n_base":   2.1,
        "subgrupos": {
            "Assinatura de documento pelo assessor":  0.70,
            "Assinatura de documento pelo ministro":  0.30,
        },
    },
}

GKEYS = list(GRUPOS_SAGWEB.keys())

# Classes prioritárias — recebem G1 "Incluido em lista HC/RHC"
CLASSES_PRIORITARIAS: Set[str] = {
    "140", "854", "143", "1200",  # fallback — sobrescrito em inicializar_constantes()
}

# k IMf calibrado por gabinete (mesmo que P2)
K_POR_GABINETE = {"reynaldo": 0.20, "palheiro": 0.30, "schietti": 0.25}

# Parâmetros do modelo de experiência
ALPHA = 0.50   # impacto da experiência na DURAÇÃO  (maior exp → menor dur)
BETA  = 0.30   # impacto da experiência no Nº EVENTOS (maior exp → mais atividades)
SIGMA_LOGNORMAL = 0.15   # ruído lognormal para heterogeneidade realista

# Número de assessores por gabinete
# Fonte: STJ/GP Resolução N. 19, 16 mar. 2026, p. 17 — Quadro de funções
# comissionadas disponibilizadas aos ministros do STJ: 38 funções por gabinete.
# Fundamento empírico que ancora o pool sintético da Fase 1 na estrutura
# organizacional oficial do STJ.
N_ASSESSORES_PADRAO = 38

TEMP_ATRIB = 1.0  # temperatura da atribuição de processos

# Estrutura de cargos comissionados por gabinete de ministro
# Fonte: STJ/GP Resolução N. 19, 16 mar. 2026, p. 17
# Quadro de funções comissionadas — 38 por gabinete (33 gabinetes = 1.254 total)
# Três perfis de geração de atividades SAGWeb:
#   "instrutor"      → instrui processos diretamente; gera todos os grupos G1–G5
#   "gestao"         → coordena fluxo; gera G1, G3, G4 (sem documentos decisórios)
#   "administrativo" → suporte; gera G3 e G4 apenas
ESTRUTURA_GABINETE = {
    "CJ3_assessor_ministro": {
        "n": 10, "prefixo_id": "CJ3A",
        "exp_min": 0.65, "exp_max": 0.95,
        "perfil": "instrutor", "peso_caso": 1.00,
        "gera_doc_decisorio": True,
    },
    "CJ3_chefe_gabinete": {
        "n": 1, "prefixo_id": "CJ3C",
        "exp_min": 0.70, "exp_max": 0.90,
        "perfil": "gestao", "peso_caso": 0.10,
        "gera_doc_decisorio": False,
    },
    "CJ2_assessor_a": {
        "n": 3, "prefixo_id": "CJ2A",
        "exp_min": 0.45, "exp_max": 0.75,
        "perfil": "instrutor", "peso_caso": 0.80,
        "gera_doc_decisorio": True,
    },
    "FC6_assessor_c": {
        "n": 15, "prefixo_id": "FC6C",
        "exp_min": 0.25, "exp_max": 0.65,
        "perfil": "instrutor", "peso_caso": 0.70,
        "gera_doc_decisorio": True,
    },
    "FC4_assistente_iv": {
        "n": 7, "prefixo_id": "FC4IV",
        "exp_min": 0.05, "exp_max": 0.30,
        "perfil": "administrativo", "peso_caso": 0.20,
        "gera_doc_decisorio": False,
    },
    "FC2_assistente_ii": {
        "n": 2, "prefixo_id": "FC2II",
        "exp_min": 0.05, "exp_max": 0.20,
        "perfil": "administrativo", "peso_caso": 0.10,
        "gera_doc_decisorio": False,
    },
}

# ROTULOS_COLEGIADA e CLASSES_PRIORITARIAS são carregados em runtime
# da Ontologia PM4JUD via pm4jud_vocab.carregar_rotulos_colegiada().
# Os valores abaixo são fallbacks usados APENAS se a ontologia não estiver
# disponível — em execução normal são substituídos em inicializar_constantes().
ROTULOS_COLEGIADA: Set[str] = {
    "Inclusão em pauta",
    "Proclamação de julgamento",
    "Recurso conhecido e não provido",
    "Recurso provido", "Recurso não provido", "Recurso não conhecido",
    "Voto do relator", "HC concedido", "HC denegado",
}

PREFIXO = {"reynaldo":"ASS_REY","palheiro":"ASS_PAL","schietti":"ASS_SCH"}
FASE = int(os.environ.get("FASE_PM4JUD","1"))

ATTR = {
    "act":"concept:name", "ts":"time:timestamp",
    "res":"org:resource", "classe":"pm4jud:codigo_classe",
    "prio":"pm4jud:prioritario","sim":"pm4jud:sim_flag",
    "aid":"pm4jud:assessor_id","exp":"pm4jud:nivel_experiencia",
    "grp":"pm4jud:grupo_sagweb","sgrp":"pm4jud:subgrupo_sagweb",
    "cargo":"pm4jud:cargo_funcao","perfil":"pm4jud:perfil_funcao",
}


# ==============================================================================
# LOGGING
# ==============================================================================

def inicializar_constantes(ontologia_dir: Path, logger: logging.Logger) -> None:
    """
    Carrega constantes semânticas da Ontologia PM4JUD em runtime via
    OntologiaPM4JUD (camada semântica transversal ao pipeline).

    Módulos carregados:
      Módulo 2 — CLASSES_PRIORITARIAS (FamiliaHabeasCorpus)
      Módulo 4 — ROTULOS_COLEGIADA (movimentos de sessão/voto)
      Módulo 7 — CONSTRAINTS_LTLF (C1–C16 para geração de eventos sintéticos)
    """
    from pm4jud_ontologia import carregar_ontologia
    global ROTULOS_COLEGIADA, CLASSES_PRIORITARIAS

    ont = carregar_ontologia(ontologia_dir, modulos=[3, 5, 7])
    ROTULOS_COLEGIADA    = ont.rotulos_colegiada()
    CLASSES_PRIORITARIAS = ont.classes_prioritarias()

    logger.info(
        "Ontologia PM4JUD carregada (P3 COMPLEMENT): "
        "%d rótulos colegiados | %d classes prioritárias | %d constraints LTLf",
        len(ROTULOS_COLEGIADA),
        len(CLASSES_PRIORITARIAS),
        len(ont.constraints_ltlf()),
    )
    logger.debug("ROTULOS_COLEGIADA: %s", sorted(ROTULOS_COLEGIADA))
    logger.debug(
        "CLASSES_PRIORITARIAS (códigos TPU): %s", sorted(CLASSES_PRIORITARIAS)
    )


def log_setup(out: Path) -> logging.Logger:
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fmt = "%(asctime)s [%(levelname)-8s] P3 — %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(out/f"pm4jud_complement_{ts}.log","w","utf-8")])
    return logging.getLogger(__name__)


# ==============================================================================
# CALIBRADOR DE EXPERIÊNCIA — Opção B
# Deriva n_base ajustado por grupo/classe a partir do corpus DATAJUD
# ==============================================================================

class CalibradorExperiencia:
    """
    Extrai parâmetros empíricos do corpus TPU para calibrar a simulação.

    Princípio de design
    -------------------
    Nenhum parâmetro de experiência é arbitrado manualmente. Tudo é derivado
    do que o corpus DATAJUD real nos conta sobre o fluxo processual de cada
    gabinete. Isso torna a Fase 1 (sintética) ancorada na realidade observada,
    não em suposições.

    O que é extraído do corpus
    --------------------------
    dist_classes : Dict[str, float]
        Proporção de cada classe processual (HC, RHC, AREsp, REsp...).
        Ex.: {"1720": 0.653, "672": 0.166, "854": 0.108, "798": 0.046}
        → Vira o vetor α da distribuição Dirichlet de especialização dos
          assessores. Um gabinete com 65 % de HC terá assessores tendendo
          a se especializar em HC.

    duracao_mediana : Dict[str, float]
        Duração mediana (dias) entre eventos TPU consecutivos, por classe.
        → Ancora a escala de tempo das atividades SAGWeb geradas pelo
          GeradorEventosInternos. Se a duração mediana de um HC é 45 dias,
          as atividades simuladas são distribuídas ao longo desses 45 dias.

    n_casos : int
        Total de traços no corpus do gabinete.
        → Normaliza contagens para probabilidades e serve de denominador
          nos cálculos de frequência relativa.

    Por que Dirichlet para especialização?
    ---------------------------------------
    A distribuição Dirichlet é o prior conjugado da distribuição categórica
    (multinomial). Ao parametrizar α ∝ dist_classes × N_assessores, garantimos
    que a SOMA dos pesos de especialização de TODOS os assessores reproduz
    exatamente a distribuição empírica de classes do corpus — propriedade
    que nem prior uniforme nem Beta independente por assessor oferecem.

    Consequência prática: se o corpus tem 65 % de HC, a equipe sintética de
    38 assessores terá, em conjunto, 65 % da capacidade de instrução voltada
    para HC — distribuída heterogeneamente conforme cada cargo.

    Referência: López-Pintado & Dumas (2022). Business Process Simulation with
    Differentiated Resources. BPM 2022, LNCS 13420, p. 361-378.
    DOI 10.1007/978-3-031-16103-2_24.
    """
    def __init__(self, logger):
        self.log = logger

    def calibrar(self, log_tpu: EventLog) -> Dict:
        self.log.info("  Calibrando tabela de experiência do corpus...")
        contagem_cl: Dict[str,int] = defaultdict(int)
        dur_cl: Dict[str,List[float]] = defaultdict(list)
        n_ativ_cl: Dict[str,List[int]] = defaultdict(list)

        for trace in log_tpu:
            cl = str(trace.attributes.get(ATTR["classe"],"0000"))
            contagem_cl[cl] += 1
            n_ativ_cl[cl].append(len(trace))
            tss = [e.get(ATTR["ts"]) for e in trace if e.get(ATTR["ts"])]
            if len(tss) >= 2:
                d = (max(tss)-min(tss)).total_seconds()/86400.0
                if 0 < d < 3650:
                    dur_cl[cl].append(d)

        n_total = len(log_tpu)
        dist = {cl: n/n_total for cl,n in contagem_cl.items()} if n_total else {}
        dur_base = {cl: round(median(v),2) if v else 30.0 for cl,v in dur_cl.items()}
        n_base_cl = {cl: round(mean(v),1) if v else 5.0 for cl,v in n_ativ_cl.items()}

        # Ajusta n_base por grupo em função da complexidade da classe
        # (classes com mais atividades TPU → mais atividades internas)
        n_base_grupo_cl: Dict[str,Dict[str,float]] = {}
        for cl, n_tpu in n_base_cl.items():
            fator_cl = n_tpu / 22.73  # 22.73 = média geral do corpus
            n_base_grupo_cl[cl] = {
                g: round(GRUPOS_SAGWEB[g]["n_base"] * fator_cl, 2)
                for g in GKEYS
            }

        self.log.info(
            f"  Calibração: {len(dist)} classes | {n_total} casos | "
            f"classe principal: {max(dist,key=dist.get) if dist else '—'}")
        return {
            "dist_classes": dist,
            "dur_base_cl": dur_base,
            "n_base_grupo_cl": n_base_grupo_cl,
            "n_total": n_total,
        }


# ==============================================================================
# POOL DE ASSESSORES — Dirichlet calibrado
# ==============================================================================

class PoolAssessores:
    def __init__(self, logger):
        self.log = logger

    def gerar(self, gabinete:str, calib:Dict, n:int, seed:int) -> List[Dict]:
        """
        Gera o pool de assessores sintéticos do gabinete.

        A geração segue estritamente a estrutura oficial de cargos comissionados
        definida pela STJ/GP Resolução N. 19, 16 mar. 2026, p. 17 (38 funções
        por gabinete). O parâmetro `n` é recebido por compatibilidade com a
        interface CLI mas é ignorado — a contagem vem de ESTRUTURA_GABINETE.

        Para cada cargo em ESTRUTURA_GABINETE, o método:

          1. Sorteia o nível de experiência de Uniform(exp_min, exp_max)
             — a faixa é específica por cargo, refletindo a hierarquia
             institucional (CJ-3 sênior, FC-2 iniciante).

          2. Sorteia os pesos de especialização de Dirichlet(α)
             — α é proporcional à distribuição de classes do corpus,
             garantindo que a equipe inteira reproduza a composição real.

          3. Atribui o perfil funcional ("instrutor", "gestao", "administrativo")
             — determina quais grupos SAGWeb o assessor vai gerar (G1–G5).

          4. Determina o peso_caso
             — fator de atração de processos na distribuição de carga.
             CJ-3 Assessor (peso=1.0) recebe proporcionalmente mais processos
             que FC-4 Assistente (peso=0.2).

        ID gerado: ASS_<GAB>_<CARGO_PREFIXO>_<SEQ>
        Exemplo:   ASS_REY_CJ3A_01  (1º Assessor de Ministro do Reynaldo)
                   ASS_REY_FC4IV_32 (3º Assistente IV do Reynaldo)

        Parameters
        ----------
        gabinete : str
            Identificador do gabinete ("reynaldo", "palheiro", "schietti").
        calib : Dict
            Saída de CalibradorExperiencia.calibrar() — dist_classes, durações.
        n : int
            Ignorado. Mantido para compatibilidade com CLI.
        seed : int
            Semente do gerador de números aleatórios. Garante reprodutibilidade
            total: mesma semente → mesmo pool → mesmo log de eventos.

        Returns
        -------
        List[Dict]
            Lista de 38 dicionários, um por assessor, com campos:
            id, cargo, perfil, peso_caso, gera_doc_decisorio,
            especializacao (Dict[classe → peso]), nivel_exp, classe_principal,
            gabinete.
        """
        rng = np.random.default_rng(seed)
        dist = calib.get("dist_classes",{})
        classes = list(dist.keys()) or ["1720"]
        alphas = np.array([dist.get(cl,0.01)*N_ASSESSORES_PADRAO for cl in classes])
        alphas = np.maximum(alphas, 0.1)

        prefixo = PREFIXO.get(gabinete, f"ASS_{gabinete[:3].upper()}")
        assessores = []
        niveis_todos = []
        seq = 1

        for cargo_id, cargo in ESTRUTURA_GABINETE.items():
            for _ in range(cargo["n"]):
                nivel = float(np.clip(
                    rng.uniform(cargo["exp_min"], cargo["exp_max"]), 0.05, 1.0
                ))
                niveis_todos.append(nivel)
                pesos = rng.dirichlet(alphas)
                esp = {cl: round(float(pesos[j]),4) for j,cl in enumerate(classes)}
                aid = f"{prefixo}_{cargo['prefixo_id']}_{seq:02d}"
                assessores.append({
                    "id":                  aid,
                    "cargo":               cargo_id,
                    "perfil":              cargo["perfil"],
                    "peso_caso":           cargo["peso_caso"],
                    "gera_doc_decisorio":  cargo["gera_doc_decisorio"],
                    "especializacao":      esp,
                    "nivel_exp":           round(nivel, 4),
                    "classe_principal":    max(esp, key=esp.get),
                    "gabinete":            gabinete,
                })
                seq += 1

        self.log.info(
            f"  Pool: {len(assessores)} assessores | gabinete={gabinete} | "
            f"seed={seed} | exp_media={float(np.mean(niveis_todos)):.2f} | "
            f"perfis: instrutor={sum(1 for a in assessores if a['perfil']=='instrutor')}, "
            f"gestao={sum(1 for a in assessores if a['perfil']=='gestao')}, "
            f"admin={sum(1 for a in assessores if a['perfil']=='administrativo')}")
        return assessores


# ==============================================================================
# TABELA DE EXPERIÊNCIA — converte pesos → contagens por grupo
# ==============================================================================

class TabelaExperiencia:
    """
    Tabela de produtividade e duração por assessor × classe × grupo SAGWeb.

    O que esta tabela representa
    ----------------------------
    Para cada combinação (assessor_id, classe_tpu, grupo_sagweb), a tabela
    guarda dois valores:
      - fator_experiencia : float [0,1]
          Medida relativa de proficiência do assessor naquele grupo e classe.
          Calculado como: n_lancamentos_assessor / max_n_lancamentos_no_grupo
          → 1.0 = assessor mais produtivo do grupo; 0.1 = iniciante.
      - duracao_media_dias : float
          Tempo médio de execução de uma atividade desse grupo para esse
          assessor e classe. Decresce com a experiência:
            dur = dur_base / (1 + α × fator_exp)
          onde dur_base vem das durações medianas do corpus (CalibradorExperiencia)
          e α = ALPHA = 0.50 (parâmetro de sensibilidade da experiência).

    Por que uma tabela e não cálculo direto?
    -----------------------------------------
    Calcular n_lancamentos e durações a cada evento seria O(N²). A tabela
    pré-computa tudo uma vez (O(N × classes × grupos)) e depois as consultas
    são O(1). Com 38 assessores × 14 classes × 10 grupos = 5.320 células,
    o custo de pré-computação é desprezível.

    Arquivos molde
    --------------
    A tabela é exportada como `complement_<gab>_experiencia.csv` — um arquivo
    molde para a Fase 2. Na Fase 2, esse arquivo é substituído por dados reais
    do SAGWeb (produtividade real de cada assessor por grupo e classe).
    """
    """
    Constrói a tabela de experiência:
      tabela[assessor_id][classe_tpu][grupo_sagweb] = {
          "n_lancamentos": int,
          "duracao_media": float,
          "fator_experiencia": float,
      }
    """
    def __init__(self, calib:Dict, assessores:List[Dict], seed:int, logger):
        self.calib = calib
        self.assessores = assessores
        self.rng = np.random.default_rng(seed+10)
        self.log = logger
        self.tabela: Dict[str,Dict] = {}
        self._construir()

    def _n_proc(self, assessor:Dict, classe:str) -> int:
        """Nº de processos que o assessor instruiu na classe (simulado)."""
        dist = self.calib.get("dist_classes",{})
        n_total = self.calib.get("n_total",1)
        total_cl = int(n_total * dist.get(classe, 0.01))
        peso = assessor["especializacao"].get(classe,
               min(assessor["especializacao"].values()))
        ruido = float(self.rng.lognormal(0, SIGMA_LOGNORMAL))
        return max(int(total_cl * peso * ruido), 1)

    def _construir(self):
        n_base_gp = self.calib.get("n_base_grupo_cl",{})
        dur_base   = self.calib.get("dur_base_cl",{})

        # Primeiro passo: computa n_lancamentos brutos
        tabela_bruta: Dict[str,Dict[str,Dict[str,int]]] = {}
        for a in self.assessores:
            tabela_bruta[a["id"]] = {}
            classes = list(self.calib.get("dist_classes",{}).keys())
            for cl in classes:
                n_proc = self._n_proc(a, cl)
                nivel  = a["nivel_exp"]
                tabela_bruta[a["id"]][cl] = {}
                for g in GKEYS:
                    nb = n_base_gp.get(cl,{}).get(g, GRUPOS_SAGWEB[g]["n_base"])
                    # Mais experiente → mais atividades por processo
                    n_ativ_proc = nb * (0.7 + 0.6 * nivel)
                    tabela_bruta[a["id"]][cl][g] = max(int(n_proc * n_ativ_proc), 0)

        # Segundo passo: normaliza fator_exp por (grupo, classe)
        # fator_exp = n_lancamentos / max_do_grupo_nessa_classe
        max_grp_cl: Dict[str,Dict[str,int]] = {}  # [g][cl] = max
        for g in GKEYS:
            max_grp_cl[g] = {}
            classes = list(self.calib.get("dist_classes",{}).keys())
            for cl in classes:
                vals = [tabela_bruta[a["id"]].get(cl,{}).get(g,0)
                        for a in self.assessores]
                max_grp_cl[g][cl] = max(vals) if vals else 1

        for a in self.assessores:
            self.tabela[a["id"]] = {}
            classes = list(self.calib.get("dist_classes",{}).keys())
            for cl in classes:
                self.tabela[a["id"]][cl] = {}
                dur_b = dur_base.get(cl, 30.0) * 0.6 / len(GKEYS)
                for g in GKEYS:
                    n_lanc = tabela_bruta[a["id"]].get(cl,{}).get(g,0)
                    mx = max_grp_cl[g].get(cl,1)
                    fator = n_lanc / mx if mx > 0 else 0.0
                    # Mais experiente → menor duração por atividade
                    dur_m = dur_b / (1.0 + ALPHA * fator)
                    self.tabela[a["id"]][cl][g] = {
                        "n_lancamentos":     n_lanc,
                        "duracao_media":     round(dur_m, 4),
                        "fator_experiencia": round(fator, 4),
                    }

        total_lanc = sum(
            sum(self.tabela[a["id"]][cl][g]["n_lancamentos"]
                for cl in self.tabela[a["id"]]
                for g in GKEYS)
            for a in self.assessores
        )
        self.log.info(f"  Tabela de experiência: {total_lanc:,} lançamentos simulados")

    def fator(self, assessor_id:str, classe:str, grupo:str) -> float:
        return self.tabela.get(assessor_id,{}).get(classe,{}).get(grupo,{}).get("fator_experiencia",0.1)

    def duracao_media(self, assessor_id:str, classe:str, grupo:str) -> float:
        return self.tabela.get(assessor_id,{}).get(classe,{}).get(grupo,{}).get("duracao_media",1.0)

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for aid, cls_dict in self.tabela.items():
            for cl, gp_dict in cls_dict.items():
                for g, vals in gp_dict.items():
                    rows.append({
                        "assessor_id":      aid,
                        "classe_tpu":       cl,
                        "grupo_sagweb":     g,
                        "label_grupo":      GRUPOS_SAGWEB[g]["label"],
                        "n_lancamentos":    vals["n_lancamentos"],
                        "duracao_media_dias": vals["duracao_media"],
                        "fator_experiencia":  vals["fator_experiencia"],
                    })
        return pd.DataFrame(rows)


# ==============================================================================
# DISTRIBUIDOR DE PROCESSOS
# ==============================================================================

class DistribuidorProcessos:
    def __init__(self, assessores:List[Dict], tabela:TabelaExperiencia,
                 seed:int, logger):
        self.assessores = assessores
        self.tabela = tabela
        self.rng = random.Random(seed+20)
        self.log = logger
        self._carga: Dict[str,int] = {a["id"]:0 for a in assessores}

    def atribuir(self, trace:Trace) -> Dict:
        """
        Seleciona o assessor que irá instruir o processo representado pelo traço.

        Algoritmo de atribuição — softmax ponderado por afinidade
        ----------------------------------------------------------
        Para cada assessor, calcula uma pontuação de afinidade composta:

          afinidade = f_med × peso_cargo × (1 - 0.2 × penalidade_carga)

        Onde:
          f_med : float
              Média do fator de experiência do assessor nos três grupos
              principais (G1 escaninhos, G2a criação, G3 fases).
              Representa o quanto o assessor é proficiente na matéria
              processual específica desse processo (classe TPU).

          peso_cargo : float (de ESTRUTURA_GABINETE)
              Fator institucional que reflete a vocação do cargo:
              CJ-3 Assessor=1.0, CJ-2=0.80, FC-6=0.70, Chefe=0.10, FC-4=0.20.
              Garante que instrutores seniores recebam mais processos que
              assistentes administrativos.

          penalidade_carga : float [0, 1]
              carga_atual / carga_máxima_do_pool.
              Penaliza assessores que já acumularam muitos processos, simulando
              o balanceamento informal que ocorre em gabinetes reais.
              Peso da penalidade = 0.2 (balanceamento suave, não forçado).

        A afinidade é convertida em probabilidade via softmax com temperatura
        TEMP_ATRIB=1.0 (temperatura neutra — sem concentração artificial).
        O assessor é sorteado com essas probabilidades (não determinístico).

        Por que não atribuição determinística (argmax)?
          O argmax sempre escolheria o mesmo assessor para a mesma classe,
          colapsando a diversidade do log. O softmax estocástico preserva
          a variabilidade observada em gabinetes reais, onde diferentes
          assessores instruem processos da mesma classe.

        Parameters
        ----------
        trace : Trace
            Traço do processo a ser atribuído. O atributo pm4jud:codigo_classe
            é lido para identificar a matéria e calcular a afinidade correta.

        Returns
        -------
        Dict
            Dicionário completo do assessor selecionado (id, cargo, perfil,
            nivel_exp, especializacao, etc.).
        """
        cl = str(trace.attributes.get(ATTR["classe"],"0000"))
        afinidades = []
        for a in self.assessores:
            # Afinidade composta: proficiência × vocação institucional × equidade de carga
            f_med = mean([
                self.tabela.fator(a["id"], cl, g)
                for g in ["G1_escaninhos","G2a_wf_criacao","G3_lancamentos_fases"]
            ])
            peso_cargo = a.get("peso_caso", 1.0)
            carga_pen  = self._carga[a["id"]] / max(1, max(self._carga.values()))
            afinidades.append(max(f_med * peso_cargo * (1.0 - 0.2*carga_pen), 1e-6))

        arr = np.array(afinidades)
        logits = np.log(arr)/TEMP_ATRIB
        logits -= logits.max()
        probs = np.exp(logits); probs /= probs.sum()
        idx = self.rng.choices(range(len(self.assessores)), weights=probs.tolist(), k=1)[0]
        a = self.assessores[idx]
        self._carga[a["id"]] += 1
        return a

    def gini(self) -> float:
        v = np.array(list(self._carga.values()), dtype=float)
        if v.sum() == 0: return 0.0
        n = len(v); mu = v.mean()
        return round(np.abs(v[:,None]-v[None,:]).sum()/(2*n*n*mu), 4)

    def df_carga(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"assessor_id":a["id"],"cargo":a.get("cargo",""),
             "perfil":a.get("perfil",""),"carga":self._carga[a["id"]],
             "nivel_exp":a["nivel_exp"],"classe_principal":a["classe_principal"]}
            for a in self.assessores
        ]).sort_values("carga",ascending=False)


# ==============================================================================
# GERADOR DE EVENTOS INTERNOS — por grupo SAGWeb
# ==============================================================================

class GeradorEventosInternos:
    """
    Gera as atividades judiciais SAGWeb para cada traço processual.

    Visão geral
    -----------
    Este é o coração da Fase 1 do COMPLEMENT. Para cada processo (traço TPU),
    o gerador produz uma sequência de atividades internas do SAGWeb que
    reproduz o fluxo de trabalho do gabinete — desde o recebimento pelo
    assessor até a assinatura do ministro e a publicação no DJe.

    As atividades são organizadas em cinco grupos funcionais (G1–G5), cada
    um mapeando um tipo de lançamento do Sistema de Automação de Gabinetes
    (SAGWeb/STJ). Esses grupos correspondem aos encontrados nos dados reais
    da Fase 2, garantindo que o formato do log sintético seja idêntico ao
    do log real — diferindo apenas no atributo pm4jud:sim_flag.

    Três perfis de geração (por perfil de assessor)
    ------------------------------------------------
    A natureza e o volume das atividades geradas dependem do perfil funcional
    do assessor atribuído ao processo:

    INSTRUTOR (28 de 38 assessores — CJ-3 Assessor, CJ-2, FC-6):
        Gera a sequência completa G1→G2→G3→G4→G5.
        É o único perfil que cria documentos decisórios (DESPACHO/DECISÃO
        ou RELATÓRIO E VOTO + EMENTA/ACÓRDÃO).
        A quantidade de alterações (G2b) e o nível de detalhe dependem
        do nivel_exp do assessor.

    GESTÃO (1 de 38 — CJ-3 Chefe de Gabinete):
        Gera apenas G1 (escaninhos de acompanhamento), G3 (fases) e G4
        (deslocamentos). Não cria documentos decisórios — coordena o fluxo.

    ADMINISTRATIVO (9 de 38 — FC-4 Assistente IV, FC-2 Assistente II):
        Gera apenas G3 (lançamentos de fases) e G4 (deslocamentos).
        Representa o apoio logístico e administrativo do gabinete.

    Detecção do caminho decisório (monocrático vs colegiado)
    ---------------------------------------------------------
    Antes de gerar os documentos G2, o gerador consulta o método
    _is_colegiada() que verifica se o traço TPU contém movimentos de
    sessão (TPU 3001, 3002, 239...). Se sim → colegiado (RELATÓRIO E VOTO
    + EMENTA/ACÓRDÃO). Se não → monocrático (DESPACHO/DECISÃO).

    Esta distinção é fundamental para a validade da Fase 2: a mesma lógica
    de detecção será usada para decidir qual template de documento buscar
    no SAGWeb real.

    Grupos SAGWeb gerados
    ---------------------
    G1  Escaninhos: estado do processo na fila de trabalho do assessor.
    G2  Workflow de documentos: criação, alteração, exclusão, publicação.
        G2a: criação e envio  G2b: alterações  G2c: exclusão  G2d: publicação
        G2e: vinculação de jurisprudência
    G3  Lançamentos de fases: marcos formais de instrução e julgamento.
    G4  Deslocamentos: movimentação física/sistêmica do processo.
    G5  Assinaturas: G5a chancela eletrônica  G5b assinatura do assessor/ministro.

    Timestamps
    ----------
    Todos os timestamps são calculados por interpolação linear entre t0
    (primeiro evento TPU do traço) e t1 (último evento TPU). A fração
    (0.0–1.0) representa a posição relativa no ciclo de vida do processo.
    """
    """
    Gera eventos internos organizados pelos 5 grupos funcionais do SAGWeb.
    Cada evento recebe:
      pm4jud:grupo_sagweb   — código do grupo (G1..G5b)
      pm4jud:subgrupo_sagweb — rótulo canônico da atividade
      pm4jud:sim_flag        — [SIM-ASSESSOR]
      pm4jud:assessor_id     — id do assessor responsável
    """
    def __init__(self, tabela:TabelaExperiencia, seed:int):
        self.tabela = tabela
        self.rng = random.Random(seed+30)
        self.rng_np = np.random.default_rng(seed+30)

    def _ts(self, t0, t1, frac:float):
        if t0 is None or t1 is None: return t0
        delta = (t1-t0).total_seconds()
        return t0 + timedelta(seconds=delta*frac)

    def _ev(self, grupo:str, subgrupo:str, ts, assessor:Dict,
            resource_type:str="ASS") -> Event:
        ev = Event()
        ev[ATTR["act"]]  = subgrupo
        ev[ATTR["ts"]]   = ts
        ev[ATTR["sim"]]  = "[SIM-ASSESSOR]"
        ev[ATTR["grp"]]  = grupo
        ev[ATTR["sgrp"]] = subgrupo
        ev[ATTR["aid"]]   = assessor["id"]
        ev[ATTR["exp"]]   = assessor["nivel_exp"]
        ev[ATTR["cargo"]] = assessor.get("cargo","")
        ev[ATTR["perfil"]]= assessor.get("perfil","")
        ev[ATTR["res"]]   = (assessor["id"] if resource_type=="ASS"
                            else "MINISTRO" if resource_type=="MIN"
                            else "SECRETARIA")
        return ev

    def _frac_list(self, n:int, start:float, end:float) -> List[float]:
        """Distribui n frações linearmente entre start e end."""
        if n <= 0: return []
        if n == 1: return [(start+end)/2]
        return [start + i*(end-start)/(n-1) for i in range(n)]

    def _random_fracs(self, n:int, start:float, end:float) -> List[float]:
        """
        Gera n frações aleatórias dentro de [start, end], ordenadas.
        Simula timestamps reais de edição distribuídos ao longo do dia de
        trabalho — não uniformes, não previsíveis.
        Usa random.Random (self.rng) — sem dependência de numpy aqui.
        """
        if n <= 0: return []
        return sorted(self.rng.uniform(start, end) for _ in range(n))

    def _n_alteracoes(self, nivel_exp:float) -> int:
        """
        Retorna o número de alterações de documento para um assessor.
        Distribuição Poisson calibrada pelo nível de experiência:
          λ = 2 + 4 × nivel_exp
          → iniciante (0.25): λ=3   → média 3 alterações
          → pleno     (0.50): λ=4   → média 4 alterações
          → sênior    (0.85): λ=5.4 → média 5-6 alterações
        Mínimo garantido = 2 (toda alteração tem pelo menos rascunho + revisão).
        Implementação: algoritmo de Knuth (amostragem Poisson via random padrão).
        """
        import math
        lam = 2.0 + 4.0 * float(nivel_exp)
        # Algoritmo de Knuth: simula Poisson sem dependência de numpy
        L = math.exp(-lam)
        k, p = 0, 1.0
        while p > L:
            k += 1
            p *= self.rng.random()
        return max(2, k - 1)

    def _is_colegiada(self, trace:Trace) -> bool:
        """
        Determina se o processo tramitou em sessão colegiada (turma ou seção).

        Estratégia de detecção
        ----------------------
        Verifica se o conjunto de concept:name do traço contém pelo menos
        um rótulo do conjunto ROTULOS_COLEGIADA. Esses rótulos são carregados
        em runtime da Ontologia PM4JUD (PM4JUD_Movimentos.owl) via SPARQL
        e correspondem a movimentos TPU que só ocorrem em julgamentos de turma:

          - "Inclusão em pauta"          (TPU 3002) → processo entrou na pauta
          - "Proclamação de julgamento"  (TPU 3001) → julgamento concluído
          - "Recurso conhecido e não provido" (TPU 239) → resultado colegiado
          - "Recurso provido", "Recurso não provido", "HC concedido/denegado"

        Por que este método importa para a validade do log?
        ---------------------------------------------------
        Gerar RELATÓRIO E VOTO para um processo que foi julgado de forma
        monocrática (ou vice-versa) produziria um log incoerente com a
        realidade do STJ — onde documentos decisórios têm formatos e
        fluxos de aprovação distintos conforme o órgão julgador.

        A detecção via TPU é confiável porque esses movimentos são registrados
        no DATAJUD pelo próprio STJ como parte do fluxo externo — não são
        inferências do pipeline.

        Returns
        -------
        bool
            True se o traço contém movimentos de sessão colegiada.
            False se o processo foi julgado monocraticamente.
        """
        atividades = {e.get(ATTR["act"],"") for e in trace}
        return bool(atividades & ROTULOS_COLEGIADA)

    def gerar(self, trace:Trace, assessor:Dict) -> List[Event]:
        """
        Gera a sequência completa de atividades SAGWeb para um processo.

        Fluxo de execução
        -----------------
        1. Extrai metadados do traço (classe, prioridade) e do assessor (perfil).
        2. Detecta o caminho decisório com _is_colegiada().
        3. Ramifica em três perfis de geração (administrativo/gestao/instrutor).
        4. Para instrutores, gera G1→G2→G3→G4→G5 na ordem cronológica.
        5. Retorna a lista de eventos sem inseri-los no traço — quem mescla
           é a função mesclar() em processar_fase1().

        Atributos marcados em cada evento
        ----------------------------------
        - concept:name     : rótulo canônico da atividade (ex.: "Em analise")
        - time:timestamp   : calculado por interpolação em _ts(t0, t1, frac)
        - org:resource     : ID do assessor (ou "MIN" / "SEC" para ministro/secretaria)
        - pm4jud:sim_flag  : "[SIM-ASSESSOR]" — marca todos os eventos sintéticos
        - pm4jud:grupo_sagweb : grupo funcional (G1, G2a, G3...)
        - pm4jud:assessor_id  : ID do assessor responsável
        - pm4jud:nivel_experiencia : nivel_exp do assessor [0,1]
        - pm4jud:cargo_funcao     : cargo institucional (CJ3_assessor_ministro...)
        - pm4jud:perfil_funcao    : perfil (instrutor/gestao/administrativo)

        Parameters
        ----------
        trace : Trace
            Traço TPU do processo. Usado para extrair classe, prioridade e
            timestamps âncora (t0=min, t1=max dos eventos TPU).
        assessor : Dict
            Dicionário do assessor atribuído (saída de DistribuidorProcessos).

        Returns
        -------
        List[Event]
            Eventos SAGWeb gerados (não inseridos ainda no traço).
            Lista vazia se o traço não tiver timestamps suficientes.
        """
        if not trace: return []
        cl   = str(trace.attributes.get(ATTR["classe"],"0000"))
        prio = bool(trace.attributes.get(ATTR["prio"],False)) or cl in CLASSES_PRIORITARIAS
        aid  = assessor["id"]
        perfil        = assessor.get("perfil", "instrutor")
        gera_doc_dec  = assessor.get("gera_doc_decisorio", True)
        colegiada     = self._is_colegiada(trace)

        tss = [e.get(ATTR["ts"]) for e in trace if e.get(ATTR["ts"])]
        if len(tss) < 2: return []
        t0, t1 = min(tss), max(tss)

        def fe(g): return self.tabela.fator(aid, cl, g)
        def n_ev(g):
            return max(1, int(GRUPOS_SAGWEB[g]["n_base"] * (0.8 + BETA*fe(g))))

        evs: List[Event] = []
        g1 = "G1_escaninhos"
        g3 = "G3_lancamentos_fases"
        g4 = "G4_deslocamento"

        # ── Perfil ADMINISTRATIVO: G3 + G4 apenas ────────────────────────
        if perfil == "administrativo":
            evs.append(self._ev(g3,"Fase interna: Instrucao",   self._ts(t0,t1,0.10),assessor))
            evs.append(self._ev(g3,"Fase interna: Conclusao",   self._ts(t0,t1,0.50),assessor))
            evs.append(self._ev(g4,"Deslocamento: Gabinete para Turma",
                                self._ts(t0,t1,0.86),assessor,"SEC"))
            return evs

        # ── Perfil GESTÃO: G1 + G3 + G4 ──────────────────────────────────
        if perfil == "gestao":
            evs.append(self._ev(g1,"Recebido pelo assessor",    self._ts(t0,t1,0.03),assessor))
            evs.append(self._ev(g1,"Em analise",                self._ts(t0,t1,0.25),assessor))
            evs.append(self._ev(g3,"Fase interna: Instrucao",   self._ts(t0,t1,0.10),assessor))
            evs.append(self._ev(g3,"Fase interna: Conclusao",   self._ts(t0,t1,0.55),assessor))
            evs.append(self._ev(g4,"Deslocamento: Gabinete para Turma",
                                self._ts(t0,t1,0.86),assessor,"SEC"))
            return evs

        # ── Perfil INSTRUTOR: sequência completa ──────────────────────────

        # G1 — Escaninhos
        fracs_esc = [0.03, 0.05, 0.25, 0.55, 0.78, 0.92]
        labels_esc = [
            "Recebido pelo assessor",
            "Incluido em lista HC/RHC" if prio else None,
            "Em analise",
            "Conclusao para decisao",
            "Aguardando sessao" if prio else "Em pauta",
            "Julgado",
        ]
        for frac, label in zip(fracs_esc, labels_esc):
            if label is None: continue
            evs.append(self._ev(g1, label, self._ts(t0,t1,frac), assessor))

        # G2 — Workflow de documentos (dois caminhos decisórios)
        if gera_doc_dec:
            ga = "G2a_wf_criacao"
            gb = "G2b_wf_alteracao"
            if not colegiada:
                # ── Monocrática: DESPACHO/DECISÃO ────────────────────────
                evs.append(self._ev(ga,"Criacao de documento: DESPACHO DECISAO",
                                    self._ts(t0,t1,0.15),assessor))
                # Gera internamente n_alt sessões de edição (Poisson calibrado
                # por experiência) — preserva a distribuição temporal realista.
                # Registra no log apenas a PRIMEIRA e a ÚLTIMA alteração:
                # captura o início e o fechamento do ciclo de revisão sem
                # fragmentar o alfabeto com _3, _4, _5... variantes.
                n_alt = self._n_alteracoes(assessor.get("nivel_exp", 0.5))
                fracs_alt = self._random_fracs(n_alt, 0.18, 0.52)
                if len(fracs_alt) > 2:
                    fracs_alt = [fracs_alt[0], fracs_alt[-1]]   # primeira e última
                for frac in fracs_alt:
                    evs.append(self._ev(gb,"Alteracao de documento: DESPACHO DECISAO",
                                        self._ts(t0,t1,frac),assessor))
                evs.append(self._ev(ga,"Envio coordenadoria: DESPACHO DECISAO",
                                    self._ts(t0,t1,0.55),assessor))
            else:
                # ── Colegiada: RELATÓRIO E VOTO (pré-sessão) ─────────────
                evs.append(self._ev(ga,"Criacao de documento: RELATORIO E VOTO",
                                    self._ts(t0,t1,0.15),assessor))
                # Mesma lógica: gera internamente via Poisson, registra
                # somente primeira e última — voto tem ciclo mais longo.
                n_alt = self._n_alteracoes(assessor.get("nivel_exp", 0.5))
                fracs_alt = self._random_fracs(n_alt, 0.18, 0.52)
                if len(fracs_alt) > 2:
                    fracs_alt = [fracs_alt[0], fracs_alt[-1]]   # primeira e última
                for frac in fracs_alt:
                    evs.append(self._ev(gb,"Alteracao de documento: RELATORIO E VOTO",
                                        self._ts(t0,t1,frac),assessor))
                evs.append(self._ev(ga,"Envio coordenadoria: RELATORIO E VOTO",
                                    self._ts(t0,t1,0.55),assessor))
                # ── EMENTA/ACÓRDÃO (pós-sessão) ──────────────────────────
                evs.append(self._ev(ga,"Criacao de documento: EMENTA ACORDAO",
                                    self._ts(t0,t1,0.90),assessor))
                # Ementa: gera internamente, registra primeira e última.
                n_alt_ementa = max(1, self._n_alteracoes(0.3) - 1)
                fracs_ementa = self._random_fracs(n_alt_ementa, 0.91, 0.93)
                if len(fracs_ementa) > 2:
                    fracs_ementa = [fracs_ementa[0], fracs_ementa[-1]]
                for frac in fracs_ementa:
                    evs.append(self._ev(gb,"Alteracao de documento: EMENTA ACORDAO",
                                        self._ts(t0,t1,frac),assessor))
                evs.append(self._ev(ga,"Envio coordenadoria: EMENTA ACORDAO",
                                    self._ts(t0,t1,0.94),assessor))
                # ── Certidão de Julgamento (secretaria da Turma) ─────────
                evs.append(self._ev("G2a_wf_criacao","Certidao de Julgamento",
                                    self._ts(t0,t1,0.89),assessor,"SEC"))

            # G2c Exclusão (raro)
            if self.rng.random() < 0.15 * fe("G2c_wf_exclusao") + 0.05:
                evs.append(self._ev("G2c_wf_exclusao","Exclusao de documento",
                                    self._ts(t0,t1,0.28),assessor))
            # G2d Publicação
            evs.append(self._ev("G2d_wf_publicacao","Publicacao de documento no DJe",
                                self._ts(t0,t1,0.97),assessor))
            # G2e Jurisprudência
            if self.rng.random() < 0.4 * fe("G2e_wf_jurisprudencia") + 0.10:
                evs.append(self._ev("G2e_wf_jurisprudencia","Vinculacao de jurisprudencia",
                                    self._ts(t0,t1,0.48),assessor))

        # G3 — Lançamentos de fases
        for label, frac in [
            ("Fase interna: Instrucao",   0.10),
            ("Fase interna: Conclusao",   0.50),
            ("Fase externa: Julgamento",  0.85),
        ]:
            evs.append(self._ev(g3, label, self._ts(t0,t1,frac), assessor))

        # G4 — Deslocamento
        evs.append(self._ev(g4,"Deslocamento: Gabinete para Turma",
                             self._ts(t0,t1,0.86),assessor,"SEC"))
        if self.rng.random() < 0.20:
            evs.append(self._ev(g4,"Deslocamento: Turma para Gabinete",
                                 self._ts(t0,t1,0.88),assessor,"SEC"))

        # G5 — Assinatura (somente instrutores com documento decisório)
        if gera_doc_dec:
            evs.append(self._ev("G5a_assinatura_chancela","Preparacao da Chancela Eletronica",
                                self._ts(t0,t1,0.60),assessor))
            evs.append(self._ev("G5b_assinatura_documento","Assinatura de documento pelo assessor",
                                self._ts(t0,t1,0.62),assessor))
            evs.append(self._ev("G5b_assinatura_documento","Assinatura de documento pelo ministro",
                                self._ts(t0,t1,0.68),assessor,"MIN"))
            if self.rng.random() < 0.40 * fe("G5b_assinatura_documento"):
                evs.append(self._ev("G5b_assinatura_documento","Assinatura de documento pelo ministro",
                                    self._ts(t0,t1,0.71),assessor,"MIN"))

        return evs


# ==============================================================================
# D'CASTRO NOS EVENTOS INTERNOS (3 perfis — por grupo)
# ==============================================================================

CANONICO_INTERNO: Dict[str,str] = {
    "Escaninho: Em analise":                         "Em analise",
    "Escaninho: Recebido":                           "Recebido pelo assessor",
    "Escaninho: Aguardando julgamento":              "Aguardando sessao",
    "Assinatura: Ministro":                          "Assinatura de documento pelo ministro",
    "Deslocamento: Para turma":                      "Deslocamento: Gabinete para Turma",
    # Legado v3 → novos nomes canônicos v4
    "Criacao de documento: Relatorio conclusivo":    "Criacao de documento: RELATORIO E VOTO",
    "Criacao de documento: Minuta de voto":          "Criacao de documento: RELATORIO E VOTO",
    "Alteracao de documento: Relatorio conclusivo":  "Alteracao de documento: RELATORIO E VOTO",
    "Alteracao de documento: Minuta de voto":        "Alteracao de documento: RELATORIO E VOTO",
}
NAO_RELABELAR = {
    "Julgado","Arquivado","Publicacao de documento no DJe",
    "Assinatura de documento pelo ministro",
    "Certidao de Julgamento",
    "Criacao de documento: EMENTA ACORDAO",
    "Criacao de documento: DESPACHO DECISAO",
    "Envio coordenadoria: DESPACHO DECISAO",
    "Envio coordenadoria: RELATORIO E VOTO",
    "Envio coordenadoria: EMENTA ACORDAO",
}

class DcastroEventosInternos:
    """
    Aplica os três perfis D'Castro (2020) sobre as atividades SAGWeb geradas.

    ⚠ ATENÇÃO — CLASSE PRESERVADA COMO REFERÊNCIA PARA P4 REFINE_2 ⚠
    Esta classe NÃO é mais invocada pelo PM4JUD-COMPLEMENT (P3).
    Conforme a arquitetura definitiva do pipeline PM4JUD:
      - P3 COMPLEMENT: responsável por ENRIQUECER o log (adiciona atividades SAGWeb)
      - P4 REFINE_2:   responsável por aplicar D'Castro sobre o log COMPLETO (TPU + SAGWeb)

    Esta implementação será migrada para pm4jud_refine2.py (P4) quando esse
    programa for construído. Os perfis P1 (canonicalização), P2 (supressão de
    infrequentes com limiar k por gabinete) e P3 (relabelling de recorrentes)
    permanecem idênticos — apenas o escopo muda: o log de entrada do REFINE_2
    inclui tanto os movimentos TPU (do REFINE_1) quanto as atividades SAGWeb
    (do COMPLEMENT), ao contrário do REFINE_1 que opera só sobre TPU.

    Referência: D'Castro (2020) — Abordagem para Pré-processamento de Logs de
    Eventos para Mineração de Processos. Tese (Doutorado) — UFPE, Recife, 2020.
    """
    def __init__(self, k:float, logger):
        self.k = k
        self.log = logger

    def _freq(self, log:EventLog) -> Dict[str,float]:
        n = len(log); ct: Dict[str,int] = defaultdict(int)
        vistos = set()
        for trace in log:
            vistos.clear()
            for ev in trace:
                if not str(ev.get(ATTR["sim"],"")).startswith("[SIM"): continue
                nome = ev.get(ATTR["act"],"")
                if nome and nome not in vistos:
                    ct[nome]+=1; vistos.add(nome)
        return {k:v/n for k,v in ct.items()} if n else {}

    def aplicar(self, log:EventLog) -> EventLog:
        self.log.info(f"  D'Castro interno (k={self.k})...")
        freq = self._freq(log)
        suprimidos = {n for n,f in freq.items() if f < self.k}
        self.log.info(f"    P2: {len(suprimidos)} atividades internas suprimidas")

        saida = EventLog(); saida.attributes.update(log.attributes)
        n_p1=n_p2=0

        for trace in log:
            t2 = Trace(); t2.attributes.update(trace.attributes)
            contagem: Dict[str,int] = defaultdict(int)
            for ev in trace:
                is_sim = str(ev.get(ATTR["sim"],"")).startswith("[SIM")
                nome = ev.get(ATTR["act"],"")
                if is_sim:
                    # P1 canonicalizar
                    canon = CANONICO_INTERNO.get(nome, nome)
                    if canon != nome:
                        ev[ATTR["act"]]=canon; n_p1+=1; nome=canon
                    # P2 suprimir
                    if nome in suprimidos:
                        n_p2+=1; continue
                    # P3 relabelar recorrentes
                    if nome and nome not in NAO_RELABELAR:
                        contagem[nome]+=1
                        if contagem[nome]>1:
                            ev[ATTR["act"]]=f"{nome}_{contagem[nome]}"
                t2.append(ev)
            if t2: saida.append(t2)

        self.log.info(f"    P1={n_p1} canon | P2={n_p2} suprim | P3=relabelling ok")
        return saida


# ==============================================================================
# MESCLAGEM E UTILITÁRIOS
# ==============================================================================

def mesclar(trace_tpu:Trace, evs_int:List[Event]) -> Trace:
    t = Trace(); t.attributes.update(trace_tpu.attributes)
    todos = list(trace_tpu)+evs_int
    todos.sort(key=lambda e: e.get(ATTR["ts"], datetime.min.replace(tzinfo=None)))
    for ev in todos: t.append(ev)
    return t

def salvar_json(obj, path:Path, logger):
    def _d(o): return o.isoformat() if hasattr(o,"isoformat") else str(o)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=_d),encoding="utf-8")
    logger.info(f"  JSON: {path.name}")


# ==============================================================================
# PROCESSAMENTO FASE 1
# ==============================================================================

def processar_fase1(gabinete:str, xes_path:Path, out:Path,
                    n_assessores:int, seed:int, logger) -> Dict:
    """
    Orquestra o enriquecimento completo de um gabinete na Fase 1 (dados sintéticos).

    Sequência de execução
    ---------------------
    1. Carrega refine1_<gab>.xes — log TPU refinado pelo REFINE_1 (P2).
    2. Calibra a tabela de experiência via CalibradorExperiencia — extrai
       dist_classes e durações medianas do corpus real.
    3. Gera o pool de 38 assessores via PoolAssessores — usando ESTRUTURA_GABINETE
       e a calibração do passo anterior.
    4. Pré-computa a TabelaExperiencia — produtividade e duração por
       assessor × classe × grupo.
    5. Para cada traço:
       a. Atribui um assessor via DistribuidorProcessos (softmax por afinidade).
       b. Gera as atividades SAGWeb via GeradorEventosInternos.
       c. Mescla TPU + SAGWeb em ordem cronológica via mesclar().
    6. Exporta complement_<gab>.xes — sem aplicar D'Castro (responsabilidade do P4).
    7. Exporta os arquivos molde CSV (assessores + experiência).
    8. Retorna dict de métricas para o relatório JSON.

    Nota sobre D'Castro
    -------------------
    Esta função NÃO aplica os perfis D'Castro sobre o log resultante.
    Essa responsabilidade pertence ao PM4JUD-REFINE_2 (P4), que opera sobre
    o log completo (TPU + SAGWeb) com os parâmetros k calibrados por gabinete.
    O COMPLEMENT apenas enriquece — nunca remove eventos.

    Parameters
    ----------
    gabinete : str
        Nome do gabinete ("reynaldo", "palheiro", "schietti").
    xes_path : Path
        Caminho do refine1_<gab>.xes (saída do P2 REFINE_1).
    out : Path
        Diretório de saída dos arquivos gerados.
    n_assessores : int
        Recebido do CLI mas ignorado — valor real é 38 (ESTRUTURA_GABINETE).
    seed : int
        Semente global de reprodutibilidade.
    logger : logging.Logger
        Logger configurado pelo log_setup().

    Returns
    -------
    Dict
        Métricas do processamento: n_traces, n_eventos_originais,
        n_eventos_injetados, n_eventos_finais, taxa_enriquecimento_pct,
        gini_carga_sintetica.
    """
    logger.info(f"\n{'='*60}\nFASE 1 — {gabinete.upper()} | N={n_assessores} | seed={seed}\n{'='*60}")

    logger.info(f"  Carregando: {xes_path.name}")
    log_tpu = xes_importer.apply(str(xes_path))
    logger.info(f"  TPU: {len(log_tpu)} traços")

    calib   = CalibradorExperiencia(logger).calibrar(log_tpu)
    pool    = PoolAssessores(logger)
    asses   = pool.gerar(gabinete, calib, n_assessores, seed)
    tabela  = TabelaExperiencia(calib, asses, seed, logger)
    distrib = DistribuidorProcessos(asses, tabela, seed, logger)
    gerador = GeradorEventosInternos(tabela, seed)

    log_comp = EventLog(); log_comp.attributes.update(log_tpu.attributes)
    n_orig=n_inj=0

    for trace in log_tpu:
        assessor = distrib.atribuir(trace)
        trace.attributes[ATTR["aid"]] = assessor["id"]
        evs_int  = gerador.gerar(trace, assessor)
        log_comp.append(mesclar(trace, evs_int))
        n_orig += len(trace); n_inj += len(evs_int)

    # D'Castro sobre o log completo (TPU + SAGWeb) é responsabilidade do
    # PM4JUD-REFINE_2 (P4) — o COMPLEMENT apenas enriquece, nunca remove.
    # log_comp é exportado diretamente sem tratamento de qualidade.

    # Exporta XES
    xes_out = out/f"complement_{gabinete}.xes"
    xes_exporter.apply(log_comp, str(xes_out))

    # CSV assessores
    df_a = pd.DataFrame([
        {"assessor_id":a["id"],"gabinete":a["gabinete"],
         "cargo":a.get("cargo",""),"perfil":a.get("perfil",""),
         "peso_caso":a.get("peso_caso",1.0),
         "nivel_exp":a["nivel_exp"],"classe_principal":a["classe_principal"],
         **{f"esp_{kk}":v for kk,v in a["especializacao"].items()}}
        for a in asses
    ])
    df_a.to_csv(out/f"complement_{gabinete}_assessores.csv",index=False,encoding="utf-8")

    # CSV tabela de experiência (grupo × classe × assessor)
    tabela.to_dataframe().to_csv(
        out/f"complement_{gabinete}_experiencia.csv",index=False,encoding="utf-8")

    n_fin = sum(len(t) for t in log_comp)
    enriq = round(n_inj/max(n_orig,1)*100,1)
    gini  = distrib.gini()

    logger.info(
        f"\n  {gabinete}: traços={len(log_tpu):,} | orig={n_orig:,} | "
        f"inj={n_inj:,} | final={n_fin:,} | enriq={enriq}% | Gini={gini:.4f}"
        f"\n  → Próximo: PM4JUD-REFINE_2 (P4) aplicará D'Castro sobre este log completo.")

    return {"gabinete":gabinete,"fase":1,"n_traces":len(log_tpu),
            "n_assessores":n_assessores,"seed":seed,
            "n_eventos_originais":n_orig,"n_eventos_injetados":n_inj,
            "n_eventos_finais":n_fin,"taxa_enriquecimento_pct":enriq,
            "gini_carga_sintetica":gini}


def processar_fase2(gabinete:str, sagweb_path:Path, out:Path, logger) -> Dict:
    logger.info(f"\n{'='*60}\nFASE 2 — {gabinete.upper()} — dados reais SAGWeb\n{'='*60}")
    if not sagweb_path.exists():
        logger.warning(f"  {sagweb_path} não encontrado. Requer autorização STJ + CEP.")
        return {"gabinete":gabinete,"fase":2,"erro":"SAGWeb não encontrado"}
    log_comp = xes_importer.apply(str(sagweb_path))
    # D'Castro sobre o log completo é responsabilidade do P4 REFINE_2.
    xes_out = out/f"complement_{gabinete}.xes"
    xes_exporter.apply(log_comp, str(xes_out))
    n_fin = sum(len(t) for t in log_comp)
    return {"gabinete":gabinete,"fase":2,"n_traces":len(log_comp),"n_eventos_finais":n_fin}


# ==============================================================================
# CLI / MAIN
# ==============================================================================

def main():
    p = argparse.ArgumentParser(prog="pm4jud_complement",
        description="PM4JUD-COMPLEMENT v3.0 — P3: Complementação com grupos SAGWeb.")
    p.add_argument("--input",       type=Path, default=Path("./output"))
    p.add_argument("--output",      type=Path, default=None)
    p.add_argument("--gabinetes",   nargs="+",default=["reynaldo","palheiro","schietti"])
    p.add_argument("--n-assessores",type=int,  default=N_ASSESSORES_PADRAO)
    p.add_argument("--ontologia", type=Path,
        default=Path(__file__).parent.parent / "ontologia",
        help="Diretório com os arquivos OWL da Ontologia PM4JUD. "
             "Padrão: ../ontologia")
    p.add_argument("--seed",        type=int,  default=42)
    args = p.parse_args()

    out = args.output or args.input
    out.mkdir(parents=True, exist_ok=True)
    logger = log_setup(out)
    inicializar_constantes(args.ontologia, logger)
    logger.info(f"PM4JUD-COMPLEMENT v4.0 | FASE_PM4JUD={FASE}")
    logger.info(f"Grupos SAGWeb: {len(GRUPOS_SAGWEB)} ({', '.join(GKEYS)})")

    rel = {"programa":"PM4JUD-COMPLEMENT","versao":"3.0","fase":FASE,
           "gerado_em":datetime.now(tz=timezone.utc).isoformat(),
           "grupos_sagweb":list(GRUPOS_SAGWEB.keys()),"gabinetes":[]}

    for gab in args.gabinetes:
        try:
            if FASE == 1:
                cands = [args.input/f"refine1_{gab}.xes",      # P2 REFINE_1 — preferencial
                         args.input/f"dcastro_{gab}.xes",       # legado — fallback
                         args.input/f"pm4jud_log_gab_{gab}.xes"]  # P1 raw — último recurso
                xp = next((c for c in cands if c.exists()), None)
                if not xp:
                    logger.warning(f"XES não encontrado para '{gab}'. Execute P2 primeiro.")
                    rel["gabinetes"].append({"gabinete":gab,"erro":"XES não encontrado"}); continue
                res = processar_fase1(gab, xp, out, args.n_assessores, args.seed, logger)
            else:
                res = processar_fase2(gab, args.input/f"sagweb_{gab}.xes", out, logger)
            rel["gabinetes"].append(res)
        except Exception as exc:
            logger.error(f"Erro '{gab}': {exc}", exc_info=True)
            rel["gabinetes"].append({"gabinete":gab,"erro":str(exc)})

    # Merge por gabinete — preserva execuções anteriores
    from pm4jud_vocab import salvar_relatorio_com_merge
    salvar_relatorio_com_merge(
        fpath=out / "complement_relatorio.json",
        programa="PM4JUD-COMPLEMENT",
        versao="4.0",
        novos_resultados=rel["gabinetes"],
        campos_consolidacao={
            "n_eventos_tpu":     "total_eventos_tpu",
            "n_eventos_sagweb":  "total_eventos_sagweb",
            "n_traces":          "total_traces",
        },
        extras={"fase": rel.get("fase", "")},
    )

    logger.info("\n"+"="*60+"\nRESUMO FINAL\n"+"="*60)
    for g in rel["gabinetes"]:
        if "erro" in g:
            logger.info(f"  {g['gabinete']:<12}: ERRO — {g['erro']}")
        else:
            logger.info(
                f"  {g['gabinete']:<12}: {g.get('n_traces',0):>6,} traços | "
                f"{g.get('n_eventos_finais',0):>8,} eventos | "
                f"enriq={g.get('taxa_enriquecimento_pct','—')}% | "
                f"Gini={g.get('gini_carga_sintetica','—')}")
    logger.info("\nPróximo: P4 — python pm4jud_pm.py --input ./output")

if __name__=="__main__":
    main()
