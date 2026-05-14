#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
PM4JUD-REFINE_1  v1.0
================================================================================

Dissertação de Mestrado — PPGIa/PUCPR
Título: PM4JUD — Otimização Multiobjetivo com Mineração de Processos e
        Simulação no Contexto do Fluxo Processual em Gabinetes de Magistrado
Autor:  Luiz Claudio Soares de Almeida
Orient: Prof. Dr. Edson Emilio Scalabrin
Ano:    2026

Descrição
---------
Implementa o pré-processamento do log de eventos judiciais segundo os três
perfis de complexidade identificados por D'Castro (2020) em logs de tribunais
brasileiros, adaptados ao corpus dos gabinetes criminais da 3.ª Seção do STJ.

Os três perfis tratam origens distintas de ruído analítico:

  PERFIL 1 — Atividades afins
    Agrupa atividades com rótulos semanticamente similares em uma
    macroatividade canônica. Utiliza similaridade TF-IDF com threshold
    configurável (padrão: 0,85).
    Exemplo: "HC concedido" e "HC concedido de ofício" podem ser
    agrupados quando o nível de abstração do modelo não requer distinção
    entre as duas formas de concessão. Já "Publicado no DJe" (TPU 92) e
    "Disponibilizado no DJe" (TPU 1061) NÃO devem ser agrupados: são
    etapas sequenciais distintas do ciclo de publicação do STJ — a
    disponibilização ocorre no dia anterior à publicação e marca o
    início da contagem de prazos processuais.
    Referência: D'Castro (2020, p. 54–58); Ramos-Gutiérrez et al. (2021).

  PERFIL 2 — Atividades infrequentes
    Remove atividades que aparecem em uma fração de traços inferior ao
    limiar k (padrão: k = 0,20, alinhado ao IMf do Inductive Miner).
    Atividades infrequentes geram variantes espúrias que comprometem a
    precisão do IMf sem acrescentar poder explicativo ao modelo.
    Referência: D'Castro (2020, p. 59–62); Leemans et al. (2013).

  PERFIL 3 — Atividades recorrentes
    Relabela atividades que aparecem múltiplas vezes no mesmo traço,
    convertendo-as em instâncias numeradas (ex.: "Ato ordinatório_1",
    "Ato ordinatório_2"). Essa estratégia elimina laços espúrios no modelo
    descoberto sem perda de informação de sequência.
    Referência: D'Castro (2020, p. 63–67).

Escopo
------
  Opera EXCLUSIVAMENTE sobre Movimentos Processuais TPU extraídos do
  DATAJUD (saída do PM4JUD-ETL, P1). Não processa Atividades Judiciais
  SAGWeb — essas são tratadas pelo PM4JUD-REFINE_2 (P4), após o
  PM4JUD-COMPLEMENT (P3) enriquecer o log com os dados internos.

Pipeline
--------
  Entrada : pm4jud_log_gab_<gabinete>.xes  (saída do PM4JUD-ETL, P1)
  Saída   :
    refine1_<gabinete>.xes          — log TPU refinado por gabinete
    refine1_relatorio.json          — métricas de qualidade antes/depois
    refine1_<gabinete>_p1_afins.csv — mapeamento de atividades agrupadas
    refine1_<gabinete>_p2_suprimidas.csv — atividades suprimidas
    refine1_<gabinete>_p3_relabelling.csv — mapeamento de relabelling

Dependência
-----------
  Requer a saída do PM4JUD-ETL (P1).
  Sua saída é entrada obrigatória do PM4JUD-COMPLEMENT (P3).

Métricas de qualidade reportadas
---------------------------------
  MF1 (perc_fit_traces)  — fração de traços que fazem replay
                           perfeito no modelo IMf descoberto
  N variantes            — número de variantes distintas no log
  N atividades           — cardinalidade do alfabeto de atividades


Referências
-----------
D'CASTRO, R. J. Abordagem para Pré-processamento de Logs de Eventos
  para Mineração de Processos. 2020. Tese (Doutorado em Ciência da
  Computação) — Universidade Federal de Pernambuco, Recife, 2020.

D'CASTRO, R. J.; OLIVEIRA, J.; TERRA, R. Process Mining Discovery in
  Judicial Processes. In: BRAZILIAN CONFERENCE ON INTELLIGENT SYSTEMS
  (BRACIS), 7., 2018, São Paulo. Anais [...]. IEEE, 2018.

LEEMANS, S. J. J.; FAHLAND, D.; VAN DER AALST, W. M. P. Discovering
  Block-Structured Process Models from Event Logs Containing Infrequent
  Behaviour. In: BUSINESS PROCESS MANAGEMENT WORKSHOPS (BPM 2013).
  Lecture Notes in Business Information Processing, v. 171.
  Springer, 2013. p. 66–78.

RAMOS-GUTIÉRREZ, B. et al. Facilitating the Understanding of Event Log
  Quality Issues via an NLP-Based Methodology. In: INTERNATIONAL CONFERENCE
  ON PROCESS MINING (ICPM), 3., 2021. IEEE, 2021.

Repositório: https://github.com/luizcsalmeida/pm4jud/tree/main/refine
================================================================================
"""

# ==============================================================================
# IMPORTAÇÕES
# ==============================================================================
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import pm4py
    from pm4py.objects.log.importer.xes import importer as xes_importer
    from pm4py.objects.log.exporter.xes import exporter as xes_exporter
    from pm4py.objects.log.obj import EventLog, Trace, Event
    from pm4py.algo.discovery.inductive import algorithm as inductive_miner
    from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
    from pm4py.statistics.variants.log import get as variants_get
    PM4PY_DISPONIVEL = True
except ImportError:
    PM4PY_DISPONIVEL = False
    print("[ERRO] pm4py não encontrado. Execute: pip install pm4py", file=sys.stderr)
    sys.exit(1)

# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================

# Nomes dos gabinetes piloto — devem corresponder ao sufixo dos .xes do P1
GABINETES_PILOTO: List[str] = ["reynaldo", "palheiro", "schietti"]

# ------ Perfil 1 — Atividades afins ----------------------------------------
# Threshold de similaridade TF-IDF para agrupamento de atividades similares.
# Valor 0,85 preserva a maioria das distinções semanticamente relevantes.
# Reduzir para 0,70 para agrupamentos mais agressivos (menor número de variantes,
# menor risco de modelos espaguete, custo: perda de granularidade).
THRESHOLD_SIMILARIDADE_P1: float = 0.85

# ------ Perfil 2 — Atividades infrequentes ----------------------------------
# Fração mínima de traços em que uma atividade deve aparecer para ser mantida.
# k = 0,20 é o mesmo limiar do Inductive Miner Infrequent (IMf) de Leemans
# et al. (2013), garantindo consistência entre o pré-processamento (P2)
# e a descoberta (P4 PM).
# D'Castro (2020, p. 60): "o valor de k deve ser calibrado empiricamente para
# cada log, mas 0,20 produz resultados satisfatórios em logs judiciais."
LIMIAR_FREQUENCIA_P2: float = 0.20   # fallback global (sem calibração por gabinete)

# Limiar k calibrado empiricamente por gabinete (Mai/2026).
# Valores derivados da maximização do MF1 no corpus DATAJUD 2024.
# Fonte: pm4jud_diagnostico_tpu.py — execução sobre 32.031 processos.
#   reynaldo: k=0,20 → MF1=92,1%  |  palheiro: k=0,30 → MF1=77,5%
#   schietti: k=0,25 → MF1=81,9%
K_POR_GABINETE: Dict[str, float] = {
    "reynaldo": 0.20,
    "palheiro": 0.30,
    "schietti": 0.25,
}

# ------ Perfil 3 — Atividades recorrentes -----------------------------------
# Número mínimo de ocorrências de uma atividade dentro de um único traço para
# que ela seja considerada "recorrente" e sujeita a relabelling.
# Valor padrão 2: atividade que aparece 2 ou mais vezes no mesmo traço.
MIN_OCORRENCIAS_RECORRENTE_P3: int = 2

# Atividades a excluir do relabelling do Perfil 3.
# "Petição" é inerentemente recorrente em processos judiciais (múltiplas
# petições ao longo da tramitação) e deve ter laços preservados no modelo.
EXCLUIR_RELABELLING_P3: Set[str] = {
    "",                    # eventos sem nome — filtrados no P2, excluídos do relabelling
    "Petição",
    "Protocolo de Petição",
    "Documento",
    # Atividades terminais — laços representam casos distintos, não repetições
    # do mesmo ciclo; o relabelling destrói a estrutura de modelo para IMf
    "Baixa Definitiva",        # encerramento definitivo do processo
    "Arquivado Definitivamente", # arquivamento definitivo
    "Trânsito em julgado",     # coisa julgada — evento terminal por natureza
}

# ------ Atributos XES -------------------------------------------------------
ATTR_ATIVIDADE = "concept:name"
ATTR_TIMESTAMP = "time:timestamp"
ATTR_RESOURCE  = "org:resource"
ATTR_CASE_ID   = "concept:name"  # atributo do traço

# Atributo customizado PM4JUD injetado nos eventos pré-processados
# para rastreabilidade das transformações aplicadas
ATTR_ORIGEM_REFINE1 = "pm4jud:refine1_perfis"

# ------ IMf — parâmetros de avaliação de fitness ----------------------------
# Variant do Inductive Miner usado na avaliação de fitness pós-P2.
# IMf = Inductive Miner Infrequent com limiar k = LIMIAR_FREQUENCIA_P2.
# O fitness calculado aqui serve como critério de aprovação antes do P3.
LIMIAR_FITNESS_MINIMO: float = 0.75


# ==============================================================================
# LOGGING
# ==============================================================================

def configurar_logging(output_dir: Path) -> logging.Logger:
    """
    Configura logging dual (console + arquivo) com timestamp no nome do arquivo.

    O arquivo de log é salvo em output_dir para auditoria e reprodução
    do experimento. O formato inclui módulo e nível para facilitar a
    depuração em pipelines longos.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"pm4jud_refine1_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    logger = logging.getLogger("PM4JUD-REFINE_1")
    logger.info("Log de auditoria iniciado: %s", log_path)
    return logger


# ==============================================================================
# UTILITÁRIOS
# ==============================================================================

def extrair_atividades(log: EventLog) -> List[str]:
    """
    Retorna a lista de nomes de atividades distintos presentes no log.

    O alfabeto de atividades é o domínio sobre o qual os três perfis
    de D'Castro operam. A lista é ordenada alfabeticamente para
    reprodutibilidade da matriz TF-IDF no Perfil 1.
    """
    atividades: Set[str] = set()
    for trace in log:
        for event in trace:
            nome = event.get(ATTR_ATIVIDADE, "")
            if nome and nome.strip():  # ignora nomes vazios ou só espaço
                atividades.add(nome)
    return sorted(atividades)


def contar_ocorrencias_por_atividade(log: EventLog) -> Dict[str, int]:
    """
    Conta em quantos traços distintos cada atividade aparece ao menos uma vez.

    Retorna um dicionário {nome_atividade: n_traços}, usado pelo Perfil 2
    para identificar atividades infrequentes.

    Nota: a contagem é por traço (não por evento). Uma atividade que aparece
    5 vezes no mesmo traço contribui com 1 para sua contagem, não com 5.
    Isso alinha a métrica ao threshold k do IMf, que opera sobre variantes
    de traços, não sobre frequência absoluta de eventos.
    """
    contagem: Dict[str, int] = {}
    for trace in log:
        atividades_no_trace: Set[str] = set()
        for event in trace:
            nome = event.get(ATTR_ATIVIDADE, "")
            if nome:
                atividades_no_trace.add(nome)
        for nome in atividades_no_trace:
            if nome:  # ignora atividades com nome vazio (ruído DATAJUD)
                contagem[nome] = contagem.get(nome, 0) + 1
    return contagem


def calcular_mf1(log: EventLog, limiar_imf: float = LIMIAR_FREQUENCIA_P2) -> float:
    """
    Calcula a métrica MF1 (perc_fit_traces) via Inductive Miner Infrequent.

    MF1 é a fração de traços do log que fazem replay perfeito no modelo
    descoberto pelo IMf com o limiar k especificado. Valores abaixo de
    LIMIAR_FITNESS_MINIMO (0,75) interrompem o pipeline e acionam revisão
    dos parâmetros de filtragem.

    Nota de compatibilidade:
      A API interna do pm4py para parâmetros do IMf varia entre versões.
      Esta função tenta três formas de chamada em sequência, do mais
      específico ao mais genérico, garantindo compatibilidade com
      pm4py >= 2.7.0 até versões recentes (>= 2.11.0).

    Referência: D'Castro (2020, p. 71); Ferronato (2022, p. 34).

    Parameters
    ----------
    log : EventLog
        Log de eventos pm4py.
    limiar_imf : float
        Threshold k do IMf (0 ≤ k ≤ 1). Padrão: LIMIAR_FREQUENCIA_P2.

    Returns
    -------
    float
        Valor de MF1 em [0, 1].
    """
    if len(log) == 0:
        return 0.0

    logger = logging.getLogger("PM4JUD-REFINE_1")

    def _normalizar(valor: float) -> float:
        """
        Normaliza o MF1 para escala [0, 1].

        O pm4py retorna perc_fit_traces em escala 0–100 quando usa
        a API de alto nível (discover_petri_net_inductive) e em 0–1
        quando usa a API interna (replay_fitness). Esta função
        unifica as duas escalas dividindo por 100 quando o valor
        excede 1,0, garantindo que o limiar LIMIAR_FITNESS_MINIMO
        seja sempre comparado na mesma escala.
        """
        return valor / 100.0 if valor > 1.0 else valor

    # Tenta a API de alto nível do pm4py (>= 2.9.0) — mais estável
    try:
        import pm4py as _pm4py
        net, im, fm = _pm4py.discover_petri_net_inductive(
            log, noise_threshold=limiar_imf
        )
        fitness = replay_fitness.apply(
            log, net, im, fm,
            variant=replay_fitness.Variants.TOKEN_BASED,
        )
        return _normalizar(float(fitness.get("perc_fit_traces", 0.0)))
    except Exception as exc1:
        logger.debug("API alto nível falhou (%s). Tentando API interna...", exc1)

    # Fallback: API interna com dicionário de parâmetros genérico
    try:
        net, im, fm = inductive_miner.apply(
            log,
            variant=inductive_miner.Variants.IMf,
            parameters={"noiseThreshold": limiar_imf},
        )
        fitness = replay_fitness.apply(
            log, net, im, fm,
            variant=replay_fitness.Variants.TOKEN_BASED,
        )
        return _normalizar(float(fitness.get("perc_fit_traces", 0.0)))
    except Exception as exc2:
        logger.debug("API interna (dict) falhou (%s). Tentando sem parâmetros...", exc2)

    # Fallback final: IMf sem parâmetro explícito (usa k padrão da versão instalada)
    try:
        net, im, fm = inductive_miner.apply(
            log,
            variant=inductive_miner.Variants.IMf,
        )
        fitness = replay_fitness.apply(
            log, net, im, fm,
            variant=replay_fitness.Variants.TOKEN_BASED,
        )
        logger.warning(
            "MF1 calculado sem noise_threshold explícito "
            "(versão pm4py incompatível com parâmetro k=%.2f).", limiar_imf
        )
        return _normalizar(float(fitness.get("perc_fit_traces", 0.0)))
    except Exception as exc3:
        logger.warning("Erro ao calcular MF1: %s. Retornando 0.0.", exc3)
        return 0.0


def contar_variantes(log: EventLog) -> int:
    """Retorna o número de variantes distintas (sequências únicas) no log."""
    variantes = variants_get.get_variants(log)
    return len(variantes)


# ==============================================================================
# PERFIL 1 — ATIVIDADES AFINS
# ==============================================================================

class PerfilAtividadesAfins:
    """
    Perfil 1 de D'Castro (2020): agrupamento de atividades semanticamente afins.

    O algoritmo constrói uma matriz de similaridade TF-IDF sobre os rótulos
    das atividades do log e agrupa pares com similaridade ≥ THRESHOLD_SIMILARIDADE_P1
    sob um rótulo canônico (o mais frequente do grupo). Grupos de tamanho 1
    são mantidos sem alteração.

    Decisão de projeto — TF-IDF em vez de embeddings semânticos:
      Embeddings (e.g., BERTimbau) produziriam similaridades mais precisas
      para o português jurídico, mas introduzem dependências pesadas (torch,
      transformers) e variância entre versões de modelos, comprometendo a
      reprodutibilidade. TF-IDF sobre tokens de atividades TPU é suficiente
      porque os rótulos são curtos, controlados e não ambíguos.
      A alternativa com embeddings está documentada como trabalho futuro
      na Seção 8.4 da dissertação.

    Referência: D'Castro (2020, p. 54–58); Ramos-Gutiérrez et al. (2021).
    """

    def __init__(
        self,
        threshold: float = THRESHOLD_SIMILARIDADE_P1,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.threshold = threshold
        self.logger = logger or logging.getLogger("PM4JUD-REFINE_1.P1")
        self.mapeamento: Dict[str, str] = {}  # atividade original → canônica

    def _construir_matriz_similaridade(
        self, atividades: List[str]
    ) -> np.ndarray:
        """
        Constrói a matriz de similaridade TF-IDF (n_atividades × n_atividades).

        O analisador de tokens usa espaço e hífen como separadores, o que
        preserva distinções como "HC denegado" / "HC concedido" ao tokenizar
        "HC", "denegado" e "HC", "concedido" de forma independente.
        """
        vectorizer = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"[^\s\-]+",  # tokens separados por espaço ou hífen
            lowercase=True,
        )
        tfidf_matrix = vectorizer.fit_transform(atividades)
        return cosine_similarity(tfidf_matrix).astype(np.float32)

    @staticmethod
    def _sao_opostos_semanticos(a: str, b: str) -> bool:
        """
        Detecta se dois rótulos de atividade são semanticamente opostos.

        Cobre três padrões de oposição no domínio judicial:

        1. Pares proibidos explícitos — casos com < 2 tokens comuns mas
           semanticamente opostos (ex: HC concedido / HC denegado).

        2. Qualificadores restritivos — "em parte" e "parcialmente" qualificam
           um resultado de forma restritiva: "Julgado procedente o pedido" ≠
           "Julgado procedente em parte o pedido". Um resultado parcial é
           juridicamente distinto do resultado pleno.

        3. Negação léxica — uma das atividades contém token de negação
           ("não", "indeferida", "denegado" etc.) e a outra não, com ao
           menos 2 tokens comuns entre elas.

        Referência: Ramos-Gutiérrez et al. (2021, p. 4).
        """
        import re as _re

        # --- 1. Pares proibidos explícitos ---
        PARES_PROIBIDOS = {
            frozenset({"HC concedido",           "HC não conhecido"}),
            frozenset({"HC concedido",           "HC denegado"}),
            frozenset({"HC concedido",           "HC concedido parcialmente"}),
            frozenset({"HC denegado",            "HC concedido parcialmente"}),
            frozenset({"HC concedido de ofício", "HC não conhecido"}),
            frozenset({"Liminar deferida",       "Liminar indeferida"}),
            frozenset({"Liminar deferida",       "Liminar parcialmente deferida"}),
        }
        if frozenset({a, b}) in PARES_PROIBIDOS:
            return True

        a_lower = a.lower()
        b_lower = b.lower()
        tokens_a = set(_re.split(r"[\s\-/]+", a_lower))
        tokens_b = set(_re.split(r"[\s\-/]+", b_lower))
        comuns = tokens_a & tokens_b

        # --- 2. Qualificadores restritivos multi-token ---
        # "em parte" e "parcialmente" tornam um resultado juridicamente distinto
        QUALIFICADORES = ("em parte", "parcialmente")
        for q in QUALIFICADORES:
            if (q in a_lower) != (q in b_lower):
                if len(comuns) >= 2:
                    return True

        # --- 3. Negação léxica por token ---
        if len(comuns) < 2:
            return False
        NEGACOES = {"não", "nao", "sem", "indeferida", "indeferido",
                    "denegado", "denegada", "negar", "negado", "negada",
                    "desconhecido", "desconhecida", "improcedente",
                    "improvido", "desprovido", "desprovida"}
        neg_a = bool(tokens_a & NEGACOES)
        neg_b = bool(tokens_b & NEGACOES)
        return neg_a != neg_b

    def _extrair_grupos(
        self,
        atividades: List[str],
        matriz: np.ndarray,
    ) -> List[List[str]]:
        """
        Extrai grupos de atividades afins por single-linkage clustering,
        com proteção contra fusão de opostos semânticos.

        O clustering single-linkage é usado porque o número de grupos não é
        conhecido a priori. A proteção contra opostos detecta pares onde um
        rótulo é a negação do outro (ex: "provido" / "não-provido"), impedindo
        que similaridade lexical alta leve a fusão semanticamente incorreta.
        Referência: Ramos-Gutiérrez et al. (2021).
        """
        n = len(atividades)
        visitados = [False] * n
        grupos: List[List[int]] = []

        for i in range(n):
            if visitados[i]:
                continue
            grupo = [i]
            visitados[i] = True
            for j in range(i + 1, n):
                if visitados[j]:
                    continue
                if matriz[i, j] < self.threshold:
                    continue
                if i == j:
                    continue
                # Proteção: não fundir opostos semânticos por negação
                if self._sao_opostos_semanticos(atividades[i], atividades[j]):
                    self.logger.debug(
                        "  Proteção antinegação: '%s' e '%s' não agrupados "
                        "(similaridade=%.3f).",
                        atividades[i], atividades[j], matriz[i, j],
                    )
                    continue
                grupo.append(j)
                visitados[j] = True
            grupos.append(grupo)

        return [[atividades[idx] for idx in g] for g in grupos]

    def _escolher_canonico(self, grupo: List[str], contagem: Dict[str, int]) -> str:
        """
        Escolhe o rótulo canônico do grupo como a atividade mais frequente.

        Em caso de empate, escolhe o rótulo alfabeticamente menor para
        garantir determinismo entre execuções.
        """
        return max(grupo, key=lambda a: (contagem.get(a, 0), -len(a), a[::-1]))

    def ajustar(
        self,
        log: EventLog,
        contagem_traces: Dict[str, int],
    ) -> Dict[str, str]:
        """
        Constrói o mapeamento {atividade_original → atividade_canônica}.

        Retorna o mapeamento sem modificar o log. A aplicação ao log é feita
        pelo método `transformar`.
        """
        atividades = extrair_atividades(log)
        if len(atividades) < 2:
            self.mapeamento = {a: a for a in atividades}
            return self.mapeamento

        self.logger.info("Perfil 1: construindo matriz TF-IDF (%d atividades)...", len(atividades))
        matriz = self._construir_matriz_similaridade(atividades)
        grupos = self._extrair_grupos(atividades, matriz)

        self.mapeamento = {}
        n_agrupadas = 0
        for grupo in grupos:
            canonico = self._escolher_canonico(grupo, contagem_traces)
            for atividade in grupo:
                self.mapeamento[atividade] = canonico
            if len(grupo) > 1:
                n_agrupadas += len(grupo)
                self.logger.info(
                    "  Grupo: %s → '%s'",
                    [a for a in grupo if a != canonico],
                    canonico,
                )

        self.logger.info(
            "Perfil 1 concluído: %d atividades agrupadas em macroatividades.",
            n_agrupadas,
        )
        return self.mapeamento

    def transformar(self, log: EventLog) -> EventLog:
        """
        Aplica o mapeamento ao log, substituindo rótulos originais pelos canônicos.

        O atributo pm4jud:refine1_perfis é atualizado em cada evento
        modificado para rastreabilidade.
        """
        for trace in log:
            for event in trace:
                nome_original = event.get(ATTR_ATIVIDADE, "")
                nome_canonico = self.mapeamento.get(nome_original, nome_original)
                if nome_canonico != nome_original:
                    event[ATTR_ATIVIDADE] = nome_canonico
                    perfis_atuais = event.get(ATTR_ORIGEM_REFINE1, "")
                    event[ATTR_ORIGEM_REFINE1] = (perfis_atuais + ",P1" if perfis_atuais else "P1")
        return log

    def para_dataframe(self) -> pd.DataFrame:
        """Exporta o mapeamento para um DataFrame para gravação em CSV."""
        registros = [
            {"atividade_original": orig, "atividade_canonica": canon,
             "agrupada": orig != canon}
            for orig, canon in sorted(self.mapeamento.items())
        ]
        return pd.DataFrame(registros)


# ==============================================================================
# PERFIL 2 — ATIVIDADES INFREQUENTES
# ==============================================================================

class PerfilAtividadesInfrequentes:
    """
    Perfil 2 de D'Castro (2020): remoção de atividades infrequentes.

    Uma atividade é infrequente quando aparece em uma fração de traços inferior
    ao limiar k (padrão: LIMIAR_FREQUENCIA_P2 = 0,20). Essas atividades geram
    variantes espúrias que distorcem a matriz de footprints do IMf sem
    representar comportamento relevante do processo.

    Efeito esperado no corpus STJ:
      D'Castro (2020, p. 71) reporta que a supressão de atividades infrequentes
      elevou a métrica MF1 de < 0,01 para ≈ 0,60 em logs judiciais brasileiros.
      A combinação com os Perfis 1 e 3 elevou MF1 para ≈ 0,77.

    Referência: D'Castro (2020, p. 59–62); Leemans et al. (2013).
    """

    def __init__(
        self,
        limiar: float = LIMIAR_FREQUENCIA_P2,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.limiar = limiar
        self.logger = logger or logging.getLogger("PM4JUD-REFINE_1.P2")
        self.suprimidas: Set[str] = set()
        self.mantidas: Set[str] = set()

    def ajustar(
        self,
        log: EventLog,
        contagem_traces: Dict[str, int],
    ) -> Tuple[Set[str], Set[str]]:
        """
        Identifica atividades infrequentes e atividades a manter.

        Atividades que aparecem em menos de `limiar * len(log)` traços
        são classificadas como infrequentes e serão removidas pelo
        método `transformar`.

        Returns
        -------
        Tuple[Set[str], Set[str]]
            (atividades_suprimidas, atividades_mantidas)
        """
        n_total_traces = len(log)
        threshold_absoluto = self.limiar * n_total_traces

        self.suprimidas = set()
        self.mantidas = set()

        for atividade, n_traces in contagem_traces.items():
            if n_traces < threshold_absoluto:
                self.suprimidas.add(atividade)
                self.logger.debug(
                    "  Suprimida: '%s' (%d traços = %.1f%% < limiar %.1f%%)",
                    atividade,
                    n_traces,
                    100 * n_traces / n_total_traces,
                    100 * self.limiar,
                )
            else:
                self.mantidas.add(atividade)

        self.logger.info(
            "Perfil 2: %d atividades suprimidas (< %.0f%% dos traços), "
            "%d mantidas.",
            len(self.suprimidas),
            100 * self.limiar,
            len(self.mantidas),
        )
        return self.suprimidas, self.mantidas

    def transformar(self, log: EventLog) -> EventLog:
        """
        Remove eventos com atividades infrequentes do log.

        Traços que ficam vazios após a remoção são descartados do log.
        Traços com pelo menos 1 evento mantido são preservados, mesmo
        que tenham perdido alguns eventos intermediários — essa estratégia
        preserva a estrutura de caso e é coerente com a abordagem de
        D'Castro (2020, p. 61).
        """
        log_filtrado = EventLog()
        # Preserva atributos globais do log original
        log_filtrado.attributes.update(log.attributes)

        n_eventos_removidos = 0
        n_traces_descartados = 0

        for trace in log:
            trace_filtrado = Trace()
            trace_filtrado.attributes.update(trace.attributes)

            for event in trace:
                nome = event.get(ATTR_ATIVIDADE, "")
                # Remove eventos com nome vazio — ruído de extração do DATAJUD
                if not nome or nome in self.suprimidas:
                    n_eventos_removidos += 1
                else:
                    event_copy = Event(event)
                    perfis_atuais = event_copy.get(ATTR_ORIGEM_REFINE1, "")
                    if not perfis_atuais:
                        event_copy[ATTR_ORIGEM_REFINE1] = "P2_mantido"
                    log_filtrado._list.append if hasattr(log_filtrado, '_list') else None
                    trace_filtrado.append(event_copy)

            if len(trace_filtrado) > 0:
                log_filtrado.append(trace_filtrado)
            else:
                n_traces_descartados += 1

        self.logger.info(
            "Perfil 2 aplicado: %d eventos removidos, %d traços descartados.",
            n_eventos_removidos,
            n_traces_descartados,
        )
        return log_filtrado

    def para_dataframe(self, contagem_traces: Dict[str, int], n_total: int) -> pd.DataFrame:
        """Exporta lista de atividades suprimidas com frequências para CSV."""
        registros = [
            {
                "atividade": a,
                "n_traces": contagem_traces.get(a, 0),
                "freq_relativa": contagem_traces.get(a, 0) / max(n_total, 1),
                "limiar_k": self.limiar,
                "suprimida": a in self.suprimidas,
            }
            for a in sorted(contagem_traces.keys())
        ]
        return pd.DataFrame(registros)


# ==============================================================================
# PERFIL 3 — ATIVIDADES RECORRENTES
# ==============================================================================

class PerfilAtividadesRecorrentes:
    """
    Perfil 3 de D'Castro (2020): relabelling de atividades recorrentes.

    Atividades que aparecem múltiplas vezes em um mesmo traço criam
    laços (self-loops) no modelo descoberto pelo IMf. Quando o laço é
    espúrio — isto é, representa iterações de uma mesma ação em vez de
    um ciclo intencional do processo — o relabelling elimina o laço sem
    perder a informação de sequência.

    Estratégia de relabelling:
      Cada ocorrência de uma atividade recorrente recebe um sufixo
      numérico sequencial: "Ato ordinatório" → "Ato ordinatório_1",
      "Ato ordinatório_2", etc. O sufixo _1 é omitido na primeira
      ocorrência para manter a legibilidade do modelo.

    Atividades excluídas do relabelling (EXCLUIR_RELABELLING_P3):
      Algumas atividades são inerentemente recorrentes no domínio judicial
      e seus laços devem ser preservados no modelo. "Petição" é o exemplo
      principal: em qualquer processo, múltiplas petições são esperadas
      ao longo da tramitação e o laço representa comportamento real.

    Referência: D'Castro (2020, p. 63–67).
    """

    def __init__(
        self,
        min_ocorrencias: int = MIN_OCORRENCIAS_RECORRENTE_P3,
        excluir: Optional[Set[str]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.min_ocorrencias = min_ocorrencias
        self.excluir = excluir or EXCLUIR_RELABELLING_P3
        self.logger = logger or logging.getLogger("PM4JUD-REFINE_1.P3")
        self.atividades_recorrentes: Set[str] = set()
        self.mapa_relabelling: Dict[str, Dict[int, str]] = {}

    def ajustar(self, log: EventLog) -> Set[str]:
        """
        Identifica atividades recorrentes no log.

        Uma atividade é recorrente se aparece em pelo menos 1 traço com
        frequência ≥ MIN_OCORRENCIAS_RECORRENTE_P3 e não está em
        EXCLUIR_RELABELLING_P3.

        Returns
        -------
        Set[str]
            Conjunto de atividades sujeitas a relabelling.
        """
        # Conta frequência máxima de cada atividade dentro de um único traço
        max_freq_no_trace: Dict[str, int] = {}
        for trace in log:
            freq_no_trace: Dict[str, int] = {}
            for event in trace:
                nome = event.get(ATTR_ATIVIDADE, "")
                freq_no_trace[nome] = freq_no_trace.get(nome, 0) + 1
            for nome, freq in freq_no_trace.items():
                max_freq_no_trace[nome] = max(max_freq_no_trace.get(nome, 0), freq)

        self.atividades_recorrentes = {
            nome
            for nome, max_freq in max_freq_no_trace.items()
            if max_freq >= self.min_ocorrencias and nome not in self.excluir
        }

        # Constrói mapa de relabelling: {atividade: {ocorrência: novo_rótulo}}
        self.mapa_relabelling = {}
        for atividade in self.atividades_recorrentes:
            # Determina o número máximo de ocorrências observadas
            max_occ = max_freq_no_trace.get(atividade, 1)
            self.mapa_relabelling[atividade] = {
                i: (atividade if i == 1 else f"{atividade}_{i}")
                for i in range(1, max_occ + 1)
            }

        self.logger.info(
            "Perfil 3: %d atividades recorrentes identificadas para relabelling.",
            len(self.atividades_recorrentes),
        )
        for nome in sorted(self.atividades_recorrentes):
            self.logger.info("  '%s' → %s", nome, list(self.mapa_relabelling[nome].values()))

        return self.atividades_recorrentes

    def transformar(self, log: EventLog) -> EventLog:
        """
        Aplica o relabelling ao log.

        Percorre cada traço e renomeia as ocorrências de atividades
        recorrentes com sufixo sequencial. A ordem de renomeação segue
        a ordem cronológica dos eventos no traço (garantida pela
        ordenação por timestamp feita no P1).
        """
        for trace in log:
            # Contador de ocorrências por atividade neste traço
            contadores: Dict[str, int] = {}
            for event in trace:
                nome = event.get(ATTR_ATIVIDADE, "")
                if nome in self.atividades_recorrentes:
                    contadores[nome] = contadores.get(nome, 0) + 1
                    occ = contadores[nome]
                    novo_nome = self.mapa_relabelling[nome].get(occ, f"{nome}_{occ}")
                    if novo_nome != nome:
                        event[ATTR_ATIVIDADE] = novo_nome
                        perfis_atuais = event.get(ATTR_ORIGEM_REFINE1, "")
                        event[ATTR_ORIGEM_REFINE1] = (
                            perfis_atuais + ",P3" if perfis_atuais else "P3"
                        )
        return log

    def para_dataframe(self) -> pd.DataFrame:
        """Exporta o mapa de relabelling para DataFrame para gravação em CSV."""
        registros = []
        for atividade, mapa_occ in sorted(self.mapa_relabelling.items()):
            for occ, novo_rotulo in sorted(mapa_occ.items()):
                registros.append({
                    "atividade_original": atividade,
                    "ocorrencia_no_trace": occ,
                    "novo_rotulo": novo_rotulo,
                    "renomeada": novo_rotulo != atividade,
                })
        return pd.DataFrame(registros)


# ==============================================================================
# PROCESSADOR PRINCIPAL
# ==============================================================================

class ProcessadorRefine1:
    """
    Aplica os três perfis D'Castro (2020) sobre os Movimentos Processuais TPU.

    Posição no pipeline
    -------------------
    Entrada : pm4jud_log_gab_<gab>.xes — log TPU bruto gerado pelo ETL (P1).
              Contém EXCLUSIVAMENTE movimentos processuais do DATAJUD/CNJ.
    Saída   : refine1_<gab>.xes — log TPU refinado, pronto para o COMPLEMENT (P3).

    Escopo estrito: APENAS movimentos TPU
    --------------------------------------
    O REFINE_1 opera somente sobre os Movimentos Processuais TPU. Atividades
    Judiciais SAGWeb ainda não existem neste ponto do pipeline — elas são
    adicionadas pelo COMPLEMENT (P3). O tratamento D'Castro das atividades
    SAGWeb é feito pelo REFINE_2 (P4).

    Os três perfis D'Castro aplicados
    ------------------------------------
    Perfil 1 — Atividades afins (TF-IDF + cosine similarity)
        Agrupa movimentos TPU com nomes semanticamente equivalentes mas
        textualmente distintos. Usa TF-IDF sobre os rótulos do corpus para
        calcular similaridade coseno. Pares acima do threshold_p1 são
        unificados sob o rótulo canônico mais frequente.
        Implementado em PerfilAtividadesAfins.

    Perfil 2 — Atividades infrequentes (limiar k)
        Remove atividades que aparecem em menos de k% dos traços.
        k é calibrado por gabinete para maximizar MF1.
        Implementado em PerfilAtividadesInfrequentes.

    Perfil 3 — Atividades recorrentes (relabelling)
        Atividades TPU que se repetem dentro de um traço recebem sufixo
        sequencial (_2, _3...) para distinguir ocorrências.
        Implementado em PerfilAtividadesRecorrentes.

    Limiar k por gabinete (calibração empírica mai/2026)
    ------------------------------------------------------
    Reynaldo: k=0.20 → MF1=92,1%  (corpus: 11.395 processos)
    Palheiro: k=0.30 → MF1=77,5%  (corpus: 10.148 processos)
    Schietti: k=0.25 → MF1=81,9%  (corpus: 10.488 processos)
    Fonte: pm4jud_diagnostico_tpu.py — maximização iterativa do MF1.

    Rastreabilidade
    ---------------
    Cada trace recebe o atributo pm4jud:refine1_perfis = "PM4JUD-REFINE_1 v1.0"
    marcando que passou pelo pré-processamento D'Castro.
    """
    """
    Orquestra a aplicação sequencial dos três perfis de D'Castro (2020)
    sobre o log de um gabinete piloto.

    Sequência de aplicação:
      1. Perfil 1 (atividades afins): reduz o alfabeto de atividades
         por agrupamento semântico.
      2. Perfil 2 (atividades infrequentes): remove eventos de atividades
         raras após o agrupamento do Perfil 1.
      3. Perfil 3 (atividades recorrentes): aplica relabelling após a
         redução do Perfil 2, sobre um log já com menor variabilidade.

    A ordem importa:
      Aplicar o Perfil 1 antes do Perfil 2 garante que atividades agrupadas
      tenham sua frequência combinada avaliada, evitando que a fusão de duas
      atividades infrequentes produza uma macroatividade abaixo do limiar.
      Aplicar o Perfil 3 por último garante que o relabelling opere sobre
      o alfabeto reduzido final, não sobre atividades que serão suprimidas.
    """

    def __init__(self, output_dir: Path, logger: logging.Logger) -> None:
        self.output_dir = output_dir
        self.logger = logger
        self.relatorio: Dict[str, Any] = {}

    def processar_gabinete(
        self,
        chave: str,
        xes_path: Path,
        limiar_k: Optional[float] = None,
    ) -> Optional[EventLog]:
        """
        Executa os três perfis de refinamento sobre o log TPU de um gabinete.

        Fluxo de execução
        -----------------
        1. Carrega pm4jud_log_gab_<chave>.xes do ETL (P1).
        2. Calcula MF1 antes do refinamento (linha base de qualidade).
        3. Aplica Perfil 1 — agrupa atividades afins por TF-IDF.
        4. Aplica Perfil 2 — suprime atividades com freq < k.
        5. Aplica Perfil 3 — relabela atividades recorrentes dentro de traços.
        6. Calcula MF1 depois do refinamento.
        7. Aprova o gabinete se MF1_depois ≥ limiar_aprovação.
        8. Exporta refine1_<chave>.xes e CSVs de rastreabilidade.

        Critério de aprovação
        ---------------------
        Um gabinete é "aprovado" se o MF1 do log refinado for ≥ 0.70.
        Gabinetes reprovados interrompem o pipeline e requerem revisão
        dos parâmetros (threshold_p1 ou limiar_k) antes de prosseguir
        para o COMPLEMENT (P3).

        k efetivo (precedência)
        -----------------------
        1. Argumento limiar_k explícito (CLI --limiar-k) → máxima precedência.
        2. K_POR_GABINETE[chave] → calibração empírica por gabinete.
        3. LIMIAR_FREQUENCIA_P2 (0.20) → fallback global.

        Parameters
        ----------
        chave : str
            Identificador do gabinete ("reynaldo", "palheiro", "schietti").
        xes_path : Path
            Caminho do arquivo pm4jud_log_gab_<chave>.xes (saída do P1 ETL).
        limiar_k : float, opcional
            Sobrescreve K_POR_GABINETE se fornecido explicitamente.

        Returns
        -------
        Optional[EventLog]
            Log refinado se aprovado. None se o arquivo não existir ou
            se ocorrer erro durante o processamento.
        """
        """
        Carrega o log XES do gabinete, aplica os três perfis e exporta resultados.

        Parameters
        ----------
        chave : str
            Identificador do gabinete (ex.: "reynaldo").
        xes_path : Path
            Caminho para o arquivo .xes gerado pelo PM4JUD-ETL (P1).

        Returns
        -------
        EventLog | None
            Log pré-processado, ou None em caso de falha.
        """
        # k efetivo: parâmetro explícito > calibração por gabinete > global
        limiar_efetivo = limiar_k if limiar_k is not None else LIMIAR_FREQUENCIA_P2

        self.logger.info("=" * 70)
        self.logger.info("Processando gabinete: %s", chave.upper())
        self.logger.info("Arquivo de entrada: %s", xes_path)
        self.logger.info("k efetivo (Perfil 2): %.2f", limiar_efetivo)
        self.logger.info("=" * 70)

        # ------------------------------------------------------------------
        # 1. Carregamento do log
        # ------------------------------------------------------------------
        try:
            log_original = xes_importer.apply(str(xes_path))
        except Exception as exc:
            self.logger.error("Erro ao carregar %s: %s", xes_path, exc)
            return None

        n_traces_orig  = len(log_original)
        n_eventos_orig = sum(len(t) for t in log_original)
        n_variantes_orig = contar_variantes(log_original)
        atividades_orig  = extrair_atividades(log_original)

        self.logger.info(
            "Log carregado: %d traços | %d eventos | %d variantes | %d atividades",
            n_traces_orig, n_eventos_orig, n_variantes_orig, len(atividades_orig),
        )

        # MF1 antes do pré-processamento
        self.logger.info("Calculando MF1 (antes)...")
        mf1_antes = calcular_mf1(log_original)
        self.logger.info("MF1 (antes): %.1f%%", mf1_antes * 100)

        # Contagem de traços por atividade — usada pelos Perfis 1 e 2
        contagem_traces = contar_ocorrencias_por_atividade(log_original)

        log = log_original  # referência mutável ao longo do pipeline

        # ------------------------------------------------------------------
        # 2. Perfil 1 — Atividades afins
        # ------------------------------------------------------------------
        self.logger.info("-" * 50)
        self.logger.info("PERFIL 1 — Atividades afins (threshold=%.2f)", THRESHOLD_SIMILARIDADE_P1)
        p1 = PerfilAtividadesAfins(threshold=THRESHOLD_SIMILARIDADE_P1, logger=self.logger)
        mapeamento_p1 = p1.ajustar(log, contagem_traces)
        log = p1.transformar(log)

        # Atualiza contagem após P1 (rótulos podem ter sido fundidos)
        contagem_traces = contar_ocorrencias_por_atividade(log)

        # Exporta mapeamento P1
        df_p1 = p1.para_dataframe()
        p1_path = self.output_dir / f"refine1_{chave}_p1_afins.csv"
        df_p1.to_csv(p1_path, index=False, encoding="utf-8")
        self.logger.info("Mapeamento P1 salvo: %s", p1_path)

        # ------------------------------------------------------------------
        # 3. Perfil 2 — Atividades infrequentes
        # ------------------------------------------------------------------
        self.logger.info("-" * 50)
        self.logger.info("PERFIL 2 — Atividades infrequentes (k=%.2f)", LIMIAR_FREQUENCIA_P2)
        p2 = PerfilAtividadesInfrequentes(limiar=limiar_efetivo, logger=self.logger)
        p2.ajustar(log, contagem_traces)
        log = p2.transformar(log)

        # Exporta atividades suprimidas P2
        df_p2 = p2.para_dataframe(contagem_traces, n_traces_orig)
        p2_path = self.output_dir / f"refine1_{chave}_p2_suprimidas.csv"
        df_p2.to_csv(p2_path, index=False, encoding="utf-8")
        self.logger.info("Relatório P2 salvo: %s", p2_path)

        # ------------------------------------------------------------------
        # 4. Perfil 3 — Atividades recorrentes
        # ------------------------------------------------------------------
        self.logger.info("-" * 50)
        self.logger.info("PERFIL 3 — Atividades recorrentes (min_occ=%d)", MIN_OCORRENCIAS_RECORRENTE_P3)
        p3 = PerfilAtividadesRecorrentes(
            min_ocorrencias=MIN_OCORRENCIAS_RECORRENTE_P3,
            excluir=EXCLUIR_RELABELLING_P3,
            logger=self.logger,
        )
        p3.ajustar(log)
        log = p3.transformar(log)

        # Exporta mapa de relabelling P3
        df_p3 = p3.para_dataframe()
        p3_path = self.output_dir / f"refine1_{chave}_p3_relabelling.csv"
        df_p3.to_csv(p3_path, index=False, encoding="utf-8")
        self.logger.info("Mapeamento P3 salvo: %s", p3_path)

        # ------------------------------------------------------------------
        # 5. Métricas após pré-processamento
        # ------------------------------------------------------------------
        n_traces_pos    = len(log)
        n_eventos_pos   = sum(len(t) for t in log)
        n_variantes_pos = contar_variantes(log)
        atividades_pos  = extrair_atividades(log)

        self.logger.info("-" * 50)
        self.logger.info("Calculando MF1 (depois)...")
        mf1_depois = calcular_mf1(log)
        self.logger.info("MF1 (depois): %.1f%%", mf1_depois * 100)

        # Critério de aprovação: fitness >= LIMIAR_FITNESS_MINIMO
        if mf1_depois < LIMIAR_FITNESS_MINIMO:
            self.logger.warning(
                "⚠ MF1 (%.4f) abaixo do limiar mínimo (%.2f). "
                "Revisar parâmetros de filtragem antes de executar o P3.",
                mf1_depois, LIMIAR_FITNESS_MINIMO,
            )
        else:
            self.logger.info(
                "✓ MF1 (%.4f) ≥ limiar (%.2f). Log aprovado para o P3.",
                mf1_depois, LIMIAR_FITNESS_MINIMO,
            )

        # ------------------------------------------------------------------
        # 6. Exportação do log filtrado
        # ------------------------------------------------------------------
        xes_saida = self.output_dir / f"refine1_{chave}.xes"
        xes_exporter.apply(log, str(xes_saida))
        self.logger.info("Log filtrado exportado: %s", xes_saida)

        # ------------------------------------------------------------------
        # 7. Registro no relatório consolidado
        # ------------------------------------------------------------------
        self.relatorio[chave] = {
            "gabinete": chave,
            "xes_entrada": str(xes_path),
            "xes_saida": str(xes_saida),
            "antes": {
                "n_traces":     n_traces_orig,
                "n_eventos":    n_eventos_orig,
                "n_variantes":  n_variantes_orig,
                "n_atividades": len(atividades_orig),
                "mf1":          round(mf1_antes, 4),   # escala 0-1
            },
            "depois": {
                "n_traces":     n_traces_pos,
                "n_eventos":    n_eventos_pos,
                "n_variantes":  n_variantes_pos,
                "n_atividades": len(atividades_pos),
                "mf1":          round(mf1_depois, 4),  # escala 0-1
            },
            "perfis": {
                "p1_n_grupos_afins":        sum(1 for k, v in mapeamento_p1.items() if k != v),
                "p2_n_suprimidas":          len(p2.suprimidas),
                "p3_n_recorrentes":         len(p3.atividades_recorrentes),
            },
            "aprovado":  mf1_depois >= LIMIAR_FITNESS_MINIMO,
        }

        self.logger.info(
            "Resumo %s: traces %d→%d | variantes %d→%d | "
            "atividades %d→%d | MF1 %.4f→%.4f | aprovado: %s",
            chave,
            n_traces_orig, n_traces_pos,
            n_variantes_orig, n_variantes_pos,
            len(atividades_orig), len(atividades_pos),
            mf1_antes * 100, mf1_depois * 100,
            self.relatorio[chave]["aprovado"],
        )

        return log

    def salvar_relatorio(self) -> Path:
        """Salva refine1_relatorio.json com merge por gabinete."""
        from pm4jud_vocab import salvar_relatorio_com_merge

        # Converte self.relatorio (dict por gabinete) para lista de resultados
        novos = [
            {"gabinete": gab, **dados}
            for gab, dados in self.relatorio.items()
            if not gab.startswith("_")
        ]
        fpath = self.output_dir / "refine1_relatorio.json"
        salvar_relatorio_com_merge(
            fpath=fpath,
            programa="PM4JUD-REFINE_1",
            versao="1.0",
            novos_resultados=novos,
            campos_consolidacao={
                "n_traces_entrada": "total_traces_entrada",
                "n_traces_saida":   "total_traces_saida",
                "n_eventos_entrada":"total_eventos_entrada",
                "n_eventos_saida":  "total_eventos_saida",
            },
            extras={
                "parametros": {
                    "threshold_p1":   THRESHOLD_SIMILARIDADE_P1,
                    "limiar_k_p2":    LIMIAR_FREQUENCIA_P2,
                    "min_occ_p3":     MIN_OCORRENCIAS_RECORRENTE_P3,
                    "limiar_fitness": LIMIAR_FITNESS_MINIMO,
                }
            },
        )
        return fpath


# ==============================================================================
# INTERFACE DE LINHA DE COMANDO
# ==============================================================================

def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pm4jud_refine1",
        description=(
            "PM4JUD-REFINE_1 v1.0 — Pré-processamento do log judicial "
            "segundo os três perfis de qualidade (D'Castro, 2020)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Processa os 3 gabinetes padrão (saída em ./output)
  python pm4jud_refine1.py

  # Especifica diretório de entrada e saída
  python pm4jud_refine1.py --input ./output --output ./output/dcastro

  # Processa apenas um gabinete
  python pm4jud_refine1.py --gabinetes reynaldo

  # Ajusta threshold de similaridade P1
  python pm4jud_refine1.py --threshold-p1 0.80

  # Ajusta limiar de frequência P2
  python pm4jud_refine1.py --limiar-k 0.15
        """,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("./output"),
        help="Diretório com os .xes gerados pelo PM4JUD-ETL (P1). Padrão: ./output",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./output"),
        help="Diretório de saída para os logs filtrados e relatórios. Padrão: ./output",
    )
    parser.add_argument(
        "--gabinetes",
        nargs="+",
        default=GABINETES_PILOTO,
        choices=GABINETES_PILOTO + ["reynaldo", "palheiro", "schietti"],
        metavar="GABINETE",
        help=f"Gabinetes a processar. Padrão: {GABINETES_PILOTO}",
    )
    parser.add_argument(
        "--threshold-p1",
        type=float,
        default=THRESHOLD_SIMILARIDADE_P1,
        dest="threshold_p1",
        help=f"Threshold TF-IDF para Perfil 1 (0–1). Padrão: {THRESHOLD_SIMILARIDADE_P1}",
    )
    parser.add_argument(
        "--limiar-k",
        type=float,
        default=LIMIAR_FREQUENCIA_P2,
        dest="limiar_k",
        help=f"Limiar k para Perfil 2 (0–1). Padrão: {LIMIAR_FREQUENCIA_P2}",
    )
    parser.add_argument(
        "--ontologia",
        type=Path,
        default=Path(__file__).parent.parent / "ontologia",
        help="Diretório da Ontologia PM4JUD (7 módulos OWL/RDF). "
             "Padrão: ../ontologia",
    )
    return parser


def main() -> None:
    parser = construir_parser()
    args = parser.parse_args()

    logger = configurar_logging(args.output)
    logger.info("PM4JUD-REFINE_1 v1.0 iniciado")
    logger.info("Entrada : %s", args.input)
    logger.info("Saída   : %s", args.output)
    logger.info("Gabinetes: %s", args.gabinetes)
    logger.info(
        "Parâmetros: P1 threshold=%.2f | P2 k=%.2f | P3 min_occ=%d",
        args.threshold_p1, args.limiar_k, MIN_OCORRENCIAS_RECORRENTE_P3,
    )

    # Carrega Ontologia PM4JUD — camada semântica transversal (Módulos 2 e 4)
    # Módulo 4: ancora Perfil 1 em nomes canônicos TPU
    # Módulo 2: protege pares semanticamente distintos de agrupamento indevido
    from pm4jud_ontologia import carregar_ontologia
    ont = carregar_ontologia(args.ontologia, modulos=[3, 5])
    pares_protegidos = ont.pares_semanticamente_distintos()
    logger.info(
        "Ontologia carregada (P2 REFINE_1): %d pares semanticamente distintos protegidos.",
        len(pares_protegidos),
    )

    # Aplica parâmetros CLI aos globais
    global THRESHOLD_SIMILARIDADE_P1, LIMIAR_FREQUENCIA_P2
    THRESHOLD_SIMILARIDADE_P1 = args.threshold_p1
    LIMIAR_FREQUENCIA_P2 = args.limiar_k

    processador = ProcessadorRefine1(
        output_dir=args.output,
        logger=logger,
        pares_protegidos=pares_protegidos,  # passa pares protegidos ao processador
    )

    n_aprovados = 0
    for chave in args.gabinetes:
        # k por gabinete: CLI sobrescreve calibração apenas se valor explicitado
        k_gabinete = K_POR_GABINETE.get(chave, args.limiar_k)
        if args.limiar_k != 0.20:          # CLI explícito → tem precedência
            k_gabinete = args.limiar_k
        logger.info("  Gabinete %-12s → k=%.2f", chave, k_gabinete)

        xes_path = args.input / f"pm4jud_log_gab_{chave}.xes"
        if not xes_path.exists():
            logger.error(
                "Arquivo não encontrado: %s. "
                "Execute o PM4JUD-ETL (P1) primeiro. "
                "Padrão esperado: pm4jud_log_gab_<gabinete>.xes", xes_path
            )
            continue

        log_filtrado = processador.processar_gabinete(chave, xes_path, limiar_k=k_gabinete)
        if log_filtrado is not None and processador.relatorio.get(chave, {}).get("aprovado"):
            n_aprovados += 1

    relatorio_path = processador.salvar_relatorio()
    logger.info("=" * 70)
    logger.info(
        "PM4JUD-REFINE_1 concluído: %d/%d gabinetes aprovados (MF1 ≥ %.2f).",
        n_aprovados, len(args.gabinetes), LIMIAR_FITNESS_MINIMO,
    )
    logger.info("Relatório consolidado: %s", relatorio_path)
    logger.info(
        "Próximo passo: PM4JUD-COMPLEMENT (P3) — "
        "execute com os arquivos refine1_<gabinete>.xes como entrada."
    )

    if n_aprovados < len(args.gabinetes):
        logger.warning(
            "⚠ Gabinetes reprovados requerem revisão dos parâmetros "
            "antes de prosseguir para o PM4JUD-COMPLEMENT (P3)."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
