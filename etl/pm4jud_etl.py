#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
PM4JUD-ETL  v2.1
================================================================================

Dissertação de Mestrado — PPGIa/PUCPR
Título: PM4JUD — Otimização Multiobjetivo com Mineração de Processos e
        Simulação no Contexto do Fluxo Processual em Gabinetes de Magistrado
Autor:  Luiz Claudio Soares de Almeida
Orient: Prof. Dr. Edson Emilio Scalabrin
Ano:    2026

Descrição
---------
Pipeline ETL em duas etapas para extração de logs de eventos dos três
gabinetes criminais piloto da 3.ª Seção do STJ via API Pública DATAJUD/CNJ.

ETAPA 1 — Extração por gabinete
  Filtros aplicados na query Elasticsearch:
    • orgaoJulgador.nome.keyword  = nome exato do gabinete
    • dataAjuizamento             = jan/2024 – dez/2024
    • movimentos.codigo in [92, 1061]  → apenas processos com publicação
      (TPU 92  = Publicação | TPU 1061 = Disponibilização no DJe)
  Paginação via search_after por @timestamp (estável).
  Extrai todo o acervo disponível do período — sem limite de volume.
  Saída: raw_<gabinete>.parquet (arquivo intermediário em disco)

ETAPA 2 — Processamento e exportação
  Lê os parquet intermediários.
  Aplica fix_mojibake (correção de encoding latin1 → utf-8).
  Resolve movimentos TPU via Ontologia PM4JUD Módulo 4.
  Exporta: <gabinete>.xes (PM4Py) + pm4jud_eventos.csv +
           pm4jud_processos.csv + pm4jud_relatorio.json

Pipeline
--------
  P1 ETL -> P2 REFINE_1 -> P3 COMPLEMENT -> P4 REFINE_2
          -> [P5 PM] -> P6 LTLf -> P7a Sim2Log -> P7b DES -> P8 OPT -> P9 STAT           

Repositório: https://github.com/luizcsalmeida/pm4jud/tree/main/etl
================================================================================
"""

# ==============================================================================
# IMPORTAÇÕES
# ==============================================================================
import argparse
import re
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import pandas as pd
import requests

try:
    from rdflib import Graph
    RDFLIB_DISPONIVEL = True
except ImportError:
    RDFLIB_DISPONIVEL = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import pm4py
    from pm4py.objects.log.obj import EventLog, Trace, Event
    from pm4py.objects.log.exporter.xes import exporter as xes_exporter
    PM4PY_DISPONIVEL = True
except ImportError:
    PM4PY_DISPONIVEL = False

# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================

ENDPOINT_STJ        = "https://api-publica.datajud.cnj.jus.br/api_publica_stj/_search"
PAGE_SIZE           = 100         # DATAJUD suporta até 10.000; 100 equilibra velocidade e memória
INTERVALO_REQ       = 0.3         # segundos entre requisições
MAX_TENTATIVAS      = 5
ESPERA_INICIAL      = 2.0

DATA_INICIO_PADRAO  = "2024-01-01"
DATA_FIM_PADRAO     = "2024-12-31"
OUTPUT_DIR_PADRAO   = Path("./output")

# Movimentos TPU que indicam processo com publicação (traço completo)
# TPU 92   = Publicação
# TPU 1061 = Disponibilização no Diário da Justiça Eletrônico
CODIGOS_PUBLICACAO  = [92, 1061]

# Gabinetes piloto — 3.ª Seção STJ (Criminal)
# Nomes exatos conforme campo orgaoJulgador.nome no DATAJUD
# Fonte: pm4jud_discovery_report.txt (Teste 2), 30/04/2026
# Gabinetes piloto — 3ª Seção Criminal do STJ (5ª e 6ª Turmas)
# Fonte: pm4jud_discovery_report_v2.txt (Teste 2), 30/04/2026
GABINETES: Dict[str, Dict] = {
    "reynaldo": {
        "nome_datajud": "GABINETE DO MINISTRO REYNALDO SOARES DA FONSECA",
        "nome_completo": "Reynaldo Soares da Fonseca",
        "turma": "5ª Turma",
        "secao": "3ª Seção",
        "materia": "Direito Penal e Processual Penal",
    },
    "palheiro": {
        "nome_datajud": "GABINETE DO MINISTRO ANTONIO SALDANHA PALHEIRO",
        "nome_completo": "Antonio Saldanha Palheiro",
        "turma": "6ª Turma",
        "secao": "3ª Seção",
        "materia": "Direito Penal e Processual Penal",
    },
    "schietti": {
        "nome_datajud": "GABINETE DO MINISTRO ROGERIO SCHIETTI",
        "nome_completo": "Rogerio Schietti Cruz",
        "turma": "6ª Turma",
        "secao": "3ª Seção",
        "materia": "Direito Penal e Processual Penal",
    },
}

# Resolução TPU → rótulo XES (Ontologia PM4JUD, Módulo 4)
# Fonte: SGT/CNJ, versão 09/04/2026
MAPA_TPU: Dict[int, str] = {}  # preenchido por carregar_mapa_ontologia()

# Diretório padrão da ontologia (relativo ao arquivo ETL)
ONTOLOGIA_DIR_PADRAO = Path(__file__).parent.parent / "ontologia"
ONTOLOGIA_ARQUIVO    = "PM4JUD_Movimentos.owl"

# ==============================================================================
# CANONICAL_MAP — vocabulário controlado PM4JUD para concept:name no XES
# ==============================================================================
# Camada de canonicalização semântica aplicada APÓS a consulta à ontologia.
# Converte rótulos verbosos do stj:tipoMovimento em formas concisas, garantindo:
#   (a) distinção inequívoca entre subtipos (ex: HC concedido ≠ HC concedido parcialmente)
#   (b) estabilidade do vocabulário de atividades entre gabinetes
#   (c) compatibilidade com o limiar TF-IDF do Perfil 1 do PM4JUD-DCASTRO (k=0,85)
#
# Decisão de projeto:
#   A ontologia (Módulo 4) é a fonte autoritativa de classificação taxonômica.
#   O CANONICAL_MAP é uma política de apresentação do projeto PM4JUD — não
#   substitui a ontologia, apenas padroniza os rótulos para mineração de processos.
#   Referência: Van der Aalst (2016, p. 35) — "activity labels should be
#   unambiguous and consistent across the event log."
# ==============================================================================
# Vocabulário canônico compartilhado (P1/P3/P4/…) — única fonte de verdade
from pm4jud_vocab import CANONICAL_MAP, _limpar_tipo_movimento, carregar_rotulos_colegiada, carregar_classes_prioritarias  # noqa

# CANONICAL_MAP definido em pm4jud_vocab.py — importado acima
# Mantido aqui como referência de que o ETL usa o mapa via import
CANONICAL_MAP_LEGADO = {
    # Família Habeas Corpus — resultados (TPU 443, 447, 451, 12458, 12475)
    "Concedido o Habeas Corpus":                                    "HC concedido",
    "Denegado o Habeas Corpus":                                     "HC denegado",
    "Concedido em parte o Habeas Corpus":                           "HC concedido parcialmente",
    "Não conhecido o Habeas Corpus":                                "HC não conhecido",
    "Não conhecido o Habeas Corpus. Concedido o Habeas Corpus de ofício":
                                                                    "HC concedido de ofício",
    # Publicação / DJe (TPU 92, 1061)
    "Publicado em":                                                 "Publicado no DJe",
    "Disponibilizado no DJ Eletrônico em":                          "Disponibilizado no DJe",
    "Disponibilizado no DJ Eletrônico":                             "Disponibilizado no DJe",
    # Distribuição (TPU 26, 36, 12474)
    "Distribuído por":                                              "Distribuído",
    "Redistribuído por  em razão de":                               "Redistribuído",
    "Redistribuído por":                                            "Redistribuído",
    "Determinada a distribuição do feito":                          "Distribuição interna",
    # Conclusão e expedição (TPU 51, 60)
    "Conclusos":                                                    "Conclusão",
    "Expedição":                                                    "Mandado expedido",
    # Petição / protocolo (TPU 85, 118)
    "Juntada de Petição":                                           "Petição",
    "Juntada de Petição de":                                        "Petição",
    "Protocolizada Petição":                                        "Protocolo de Petição",
    # Remessa (TPU 123)
    "Remetidos os Autos ()":                                        "Remessa",
    # "Remetidos os Autos ()": "Remessa",  # variante já coberta acima
    # Liminar (TPU 339, 792, 892, 348)
    "Concedida a Medida Liminar":                                   "Liminar deferida",
    "Não Concedida a Medida Liminar":                               "Liminar indeferida",
    "Concedida em parte a Medida Liminar":                          "Liminar parcialmente deferida",
    "Revogada a Medida Liminar":                                    "Liminar revogada",
    # Ato ordinatório (TPU 11383)
    "Ato ordinatório praticado":                                    "Ato ordinatório",
    # Juntada genérica (TPU 581)
    "Juntada":                                                      "Documento",
    # Embargos de Declaração (TPU 198, 200, 871, 15162–15409)
    "Embargos de Declaração Acolhidos":                             "Embargos acolhidos",
    "Embargos de Declaração Não-acolhidos":                         "Embargos não acolhidos",
    "Embargos de Declaração Acolhidos em Parte":                    "Embargos acolhidos em parte",
    "Embargos de declaração acolhidos":                             "Embargos acolhidos",
    "Embargos de declaração acolhidos em parte":                    "Embargos acolhidos em parte",
    "Embargos de declaração não acolhidos":                         "Embargos não acolhidos",
    "Não conhecidos os embargos de declaração":                     "Embargos não conhecidos",
    # Voto (TPU 14093)
    "Voto do relator proferido":                                    "Voto do relator",
    # Recebimento / Julgamento — forma concisa preferida para PM
    # (ontologia usa forma longa "Recebidos os autos" / "Julgamento",
    #  mas a forma curta produz menor fragmentação do alfabeto de atividades)
    "Recebidos os autos":                                           "Recebimento",
    "Julgamento":                                                   "Julgado",
    # Redistribuição — variantes residuais após limpeza de template
    "Redistribuído por em razão":                                   "Redistribuído",
    "Redistribuído por em razão de":                                "Redistribuído",
    "Redistribuído por":                                            "Redistribuído",
    # Expedição de documento — residual após limpeza
    "Expedição de":                                                 "Mandado expedido",
    # Remetidos — variante residual
    "Remetidos os Autos ()":                                        "Remessa",
    # Trânsito em julgado — variantes (TPU 848)
    # Fonte: HC 881458/SP e HC 881499/MG — processos reais STJ mai/2026
    "Transitado em Julgado":                                        "Trânsito em julgado",
    "Transitado em julgado":                                        "Trânsito em julgado",
    # Arquivamento definitivo — variantes (TPU 246)
    "Determinado o arquivamento definitivo":                        "Arquivado definitivamente",
    "Arquivado Definitivamente":                                    "Arquivado definitivamente",

    # ── ADICIONADOS v2.1 ── Identificados em processos reais STJ mai/2026 ──────
    # HC 881458/SP (monocrática) e HC 881499/MG (monocrática → colegiada)
    # Todos os códigos presentes no SGT/CNJ (Módulo 4 da Ontologia PM4JUD).

    # Conclusos para decisão / julgamento (TPU 51)
    # Template STJ: "Conclusos para decisão ao(à) #{destinatario} (#{papel})"
    "Conclusos para decisão":                                       "Conclusão para decisão",
    "Conclusos para julgamento":                                    "Conclusão para julgamento",
    "Conclusos":                                                    "Conclusão",

    # Liminar não concedida — variante lowercase (TPU 792)
    # Template STJ: "Não concedida a medida liminar de #{parte}, #{despacho}"
    "Não concedida a medida liminar":                               "Liminar indeferida",
    "Não concedida a Medida Liminar":                               "Liminar indeferida",

    # Despacho de mero expediente (TPU 11010)
    # Template STJ: "Proferido despacho de mero expediente #{descricao}"
    "Proferido despacho de mero expediente":                        "Despacho de mero expediente",
    "Proferido despacho":                                           "Despacho de mero expediente",

    # Inclusão em pauta / mesa (TPU 3002)
    # Template STJ: "Inclusão em mesa para julgamento - #{orgao} - sessão do dia #{data}"
    "Inclusão em mesa para julgamento":                             "Inclusão em pauta",
    "Incluído em mesa para julgamento":                             "Inclusão em pauta",

    # Resultado de julgamento colegiado (TPU 239)
    # Template STJ: "Conhecido o recurso de #{parte} e não-provido,por unanimidade, pela #{orgao}"
    "Conhecido o recurso e não-provido":                            "Recurso conhecido e não provido",
    "Conhecido o recurso":                                          "Recurso conhecido",
    "Não conhecido o recurso":                                      "Recurso não conhecido",
    "Provido o recurso":                                            "Recurso provido",
    "Não provido o recurso":                                        "Recurso não provido",

    # Proclamação Final de Julgamento (TPU 3001)
    # Template STJ: "Proclamação Final de Julgamento: \"#{texto_proclamacao}\""
    # Nota: o texto livre após ":" é variável — o MAPA_TPU da ontologia
    # retorna o rótulo normalizado; este mapeamento cobre o residual.
    "Proclamação Final de Julgamento":                              "Proclamação de julgamento",
    "Proclamação de Julgamento":                                    "Proclamação de julgamento",
}



def _limpar_tipo_movimento_etl_local(tipo: str) -> str:
    """
    LEGADO — Use pm4jud_vocab._limpar_tipo_movimento() diretamente.
    Remove templates #{...} e #(...) do stj:tipoMovimento e normaliza
    o rótulo resultante para uso como concept:name no XES.

    Exemplos:
      "Concedido o Habeas Corpus a #{nome_da_parte}" → "Concedido o Habeas Corpus"
      "Publicado #{ato_publicado} em #{data}."        → "Publicado"
      "Disponibilizado no DJ Eletrônico em #(data)"  → "Disponibilizado no DJ Eletrônico"
    """
    limpo = re.sub(r"#\{[^}]+\}", "", tipo)
    limpo = re.sub(r"#\([^)]+\)", "", limpo)
    # Remove preposições/artigos isolados no final (resíduos de template)
    limpo = re.sub(
        r"\s+(a|ao|aos|de|do|da|dos|das|em|no|na|nos|nas|para|por|pelo|pela|o|os)\s*$",
        "", limpo.strip(), flags=re.IGNORECASE,
    )
    limpo = re.sub(r"\s+", " ", limpo)
    limpo = re.sub(r"[\s.,;:]+$", "", limpo.strip())
    return limpo if limpo else tipo


def carregar_mapa_ontologia(ontologia_dir: Path) -> Dict[int, str]:
    """
    Carrega o MAPA_TPU em runtime da Ontologia PM4JUD Módulo 4.

    Usa rdflib para executar consulta SPARQL sobre os indivíduos punnados
    (owl:Class + owl:NamedIndividual) da ontologia, extraindo pares
    (stj:codigoMovimentoTPU, rdfs:label@pt) para cada movimento processual.

    Técnica de punning (OWL 2 DL):
      Cada movimento é declarado simultaneamente como owl:Class (subsunção
      hierárquica na taxonomia) e como owl:NamedIndividual (instância
      referenciável no log XES). A ontologia serve como taxonomia de
      atividades e base de mapeamento semântico para o ETL.
      Referência: Lenzerini et al. (2021), Artificial Intelligence,
      v.292, p.103432, DOI 10.1016/j.artint.2020.103432.

    Parameters
    ----------
    ontologia_dir : Path
        Diretório com PM4JUD_Movimentos.owl e módulos MNI importados.

    Returns
    -------
    Dict[int, str]
        {codigo_tpu: rotulo_canonico} de todos os movimentos da ontologia.
    """
    logger = logging.getLogger("PM4JUD-ETL")

    if not RDFLIB_DISPONIVEL:
        logger.warning(
            "rdflib não disponível — MAPA_TPU vazio. "
            "Execute: pip install rdflib"
        )
        return {}

    owl_path = ontologia_dir / ONTOLOGIA_ARQUIVO
    if not owl_path.exists():
        logger.warning("Ontologia não encontrada: %s", owl_path)
        return {}

    try:
        g = Graph()
        g.parse(str(owl_path), format="xml")
        logger.info(
            "Ontologia PM4JUD_Movimentos carregada: %d triplas", len(g)
        )

            # Estratégia de consulta SPARQL
        # ---------------------------------
        # Cada movimento TPU tem dois atributos relevantes na ontologia:
        #
        #   stj:tipoMovimento : rótulo específico do STJ, com templates #{...}.
        #     Ex.: "Concedido o Habeas Corpus a #{nome_da_parte} (#{papel})"
        #     Mais específico — distingue subtipos de HC que compartilham o
        #     mesmo rdfs:label genérico ("Habeas Corpus").
        #
        #   rdfs:label : rótulo genérico da classe OWL.
        #     Ex.: "Habeas Corpus" — usado como fallback quando tipoMovimento
        #     não está disponível (movimentos mais antigos da tabela TPU).
        #
        # Preferimos tipoMovimento porque: HC concedido, HC denegado e HC não
        # conhecido compartilham rdfs:label="Habeas Corpus" mas têm semântica
        # totalmente distinta. Sem tipoMovimento, todos colapsariam no mesmo
        # concept:name — destruindo a capacidade de minerar o resultado.
        #
        # Após extrair tipoMovimento, aplicamos _limpar_tipo_movimento() para
        # remover os templates #{...} e obter o texto invariante, depois
        # aplicamos CANONICAL_MAP para padronizar variantes residuais.
        # O resultado final é o concept:name que aparecerá no log XES.
        SPARQL = """
            PREFIX stj:  <http://www.stj.jus.br/mni/intercomunicacao#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

            SELECT ?codigo ?tipo ?label WHERE {
                ?mov stj:codigoMovimentoTPU ?codigo .
                OPTIONAL { ?mov stj:tipoMovimento ?tipo }
                OPTIONAL { ?mov rdfs:label ?label . FILTER(lang(?label) = "pt") }
            }
            ORDER BY xsd:integer(?codigo)
        """

        def _limpar_tipo(tipo: str) -> str:
            """
            Remove templates #{...} e #(...) do stj:tipoMovimento e normaliza
            o rótulo resultante para uso como concept:name no XES.

            Exemplos:
              "Concedido o Habeas Corpus a #{nome_da_parte}" → "Concedido o Habeas Corpus"
              "Publicado #{ato_publicado} em #{data}."       → "Publicado"
              "Disponibilizado no DJ Eletrônico em #(data)"  → "Disponibilizado no DJ Eletrônico"
            """
            limpo = re.sub(r"#\{[^}]+\}", "", tipo)
            limpo = re.sub(r"#\([^)]+\)", "", limpo)
            # Remove preposições/artigos isolados no final (resíduos de template)
            limpo = re.sub(r"\s+(a|ao|aos|de|do|da|dos|das|em|no|na|nos|nas|para|por|pelo|pela|o|os)\s*$",
                           "", limpo.strip(), flags=re.IGNORECASE)
            # Normaliza múltiplos espaços e remove pontuação final
            limpo = re.sub(r"\s+", " ", limpo)
            limpo = re.sub(r"[\s.,;:]+$", "", limpo.strip())
            return limpo if limpo else tipo

        mapa: Dict[int, str] = {}
        for row in g.query(SPARQL):
            try:
                cod = int(row.codigo)
                # Prefere tipoMovimento (mais específico) → rdfs:label (fallback)
                if row.tipo and str(row.tipo).strip():
                    nome = _limpar_tipo_movimento(str(row.tipo))
                elif row.label and str(row.label).strip():
                    nome = str(row.label).strip()
                else:
                    continue
                if cod and nome:
                    # Aplica vocabulário controlado PM4JUD (CANONICAL_MAP)
                    mapa[cod] = CANONICAL_MAP.get(nome, nome)
            except (ValueError, AttributeError):
                continue

        logger.info(
            "MAPA_TPU: %d movimentos carregados da ontologia.", len(mapa)
        )
        return mapa

    except Exception as exc:
        logger.error("Erro ao carregar ontologia: %s", exc)
        return {}



# Classes TPU com prioridade regimental (arts. 91 e 202 RISTJ)
CLASSES_HC_PRIORITARIAS = {1720, 1013, 1722, 1064, 15224}


# ==============================================================================
# LOGGING
# ==============================================================================

def configurar_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"pm4jud_etl_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    logger = logging.getLogger("PM4JUD-ETL")
    logger.info("Log de auditoria: %s", log_path)
    return logger


# ==============================================================================
# UTILITÁRIOS
# ==============================================================================

def fix_mojibake(val: Any) -> Any:
    """
    Corrige encoding latin1→utf-8 nos strings retornados pela API DATAJUD.
    Necessário porque o Elasticsearch da API pública serializa alguns campos
    com encoding incorreto (ex: 'JÃNIOR' → 'JÚNIOR').
    Baseado em: Colab de referência PM4JUD (2026).
    """
    if isinstance(val, str):
        try:
            return val.encode("latin1").decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            return val
    elif isinstance(val, list):
        return [fix_mojibake(v) for v in val]
    elif isinstance(val, dict):
        return {k: fix_mojibake(v) for k, v in val.items()}
    return val


def resolver_atividade(codigo: int, nome: str) -> str:
    """Resolve código TPU ao rótulo canônico da Ontologia PM4JUD Módulo 4."""
    return MAPA_TPU.get(codigo, nome.strip())


# ==============================================================================
# CLIENTE DA API DATAJUD
# ==============================================================================

class ClienteDatajud:
    """
    Cliente para a API Pública do DATAJUD/CNJ (STJ).

    Implementa paginação via search_after por @timestamp,
    backoff exponencial e filtros por gabinete + publicação.

    References
    ----------
    CNJ. API Pública DATAJUD.
    Disponível em: https://datajud-wiki.cnj.jus.br/api-publica/
    Acesso em: abr. 2026.
    """

    def __init__(self, api_key: str) -> None:
        self.endpoint = ENDPOINT_STJ
        self.sessao = requests.Session()
        self.sessao.headers.update({
            "Authorization": f"ApiKey {api_key}",
            "Content-Type": "application/json",
        })
        self.logger = logging.getLogger("PM4JUD-ETL.Cliente")

    def _construir_query(
        self,
        nome_gabinete: str,
        data_inicio: str,
        data_fim: str,
        search_after: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Constrói a query Elasticsearch para extração por gabinete.

        Filtros aplicados (bool.filter — não afetam score):
          1. orgaoJulgador.nome.keyword = nome exato do gabinete
          2. dataAjuizamento no intervalo [data_inicio, data_fim]
          3. movimentos.codigo in [92, 1061] — processo com publicação

        Ordenação por @timestamp ASC para paginação estável.
        """
        query: Dict[str, Any] = {
            "size": PAGE_SIZE,
            "_source": {
                "includes": [
                    "id", "tribunal", "grau",
                    "dataAjuizamento", "numeroProcesso", "nivelSigilo",
                    "classe.codigo", "classe.nome",
                    "assuntos.codigo", "assuntos.nome",
                    "orgaoJulgador.codigo", "orgaoJulgador.nome",
                    "formato.nome", "sistema.nome",
                    "movimentos.codigo", "movimentos.nome",
                    "movimentos.dataHora", "movimentos.complementosTabelados",
                ]
            },
            "query": {
                "bool": {
                    "filter": [
                        {
                            "term": {
                                "orgaoJulgador.nome.keyword": nome_gabinete
                            }
                        },
                        {
                            "range": {
                                "dataAjuizamento": {
                                    "gte": data_inicio,
                                    "lte": data_fim,
                                    "format": "yyyy-MM-dd",
                                }
                            }
                        },
                        {
                            "terms": {
                                "movimentos.codigo": CODIGOS_PUBLICACAO
                            }
                        },
                    ]
                }
            },
            "sort": [{"@timestamp": {"order": "asc"}}],
        }
        if search_after:
            query["search_after"] = search_after
        return query

    def _requisitar(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST com backoff exponencial."""
        espera = ESPERA_INICIAL
        for tentativa in range(1, MAX_TENTATIVAS + 1):
            try:
                resp = self.sessao.post(
                    self.endpoint, json=payload, timeout=30
                )
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code if exc.response else "?"
                self.logger.warning(
                    "HTTP %s — tentativa %d/%d — aguardando %.1fs",
                    status, tentativa, MAX_TENTATIVAS, espera,
                )
            except requests.exceptions.RequestException as exc:
                self.logger.warning(
                    "Rede — tentativa %d/%d: %s — aguardando %.1fs",
                    tentativa, MAX_TENTATIVAS, exc, espera,
                )
            time.sleep(espera)
            espera = min(espera * 2, 60.0)
        raise RuntimeError(
            f"Falha após {MAX_TENTATIVAS} tentativas no endpoint {self.endpoint}"
        )

    def extrair_gabinete(
        self,
        chave: str,
        data_inicio: str,
        data_fim: str,
    ) -> List[Dict[str, Any]]:
        """
        Extrai o acervo completo do gabinete indicado no período definido,
        sem limite de volume — censo completo conforme Ferronato (2022).

        Parameters
        ----------
        chave : str
            Chave do dicionário GABINETES ('reynaldo', 'joel', 'schietti').
        data_inicio : str
            Data inicial no formato YYYY-MM-DD.
        data_fim : str
            Data final no formato YYYY-MM-DD.

        Returns
        -------
        List[Dict]
            Lista de registros brutos (campo _source da API).
        """
        gab = GABINETES[chave]
        nome_datajud = gab["nome_datajud"]
        self.logger.info(
            "Extraindo gabinete '%s' | %s | %s | %s | período: %s → %s",
            chave, gab["secao"], gab["materia"][:30],
            nome_datajud, data_inicio, data_fim,
        )

        registros: List[Dict[str, Any]] = []
        search_after = None
        pagina = 0

        while True:
            payload = self._construir_query(
                nome_datajud, data_inicio, data_fim, search_after
            )
            dados = self._requisitar(payload)
            hits = dados.get("hits", {}).get("hits", [])

            if not hits:
                self.logger.info(
                    "Gabinete '%s' — extração concluída na página %d "
                    "(%d processos).", chave, pagina, len(registros)
                )
                break

            for hit in hits:
                src = hit.get("_source", {})
                src = fix_mojibake(src)
                registros.append(src)

            pagina += 1
            search_after = hits[-1].get("sort")
            self.logger.info(
                "  Página %d | %d processos acumulados | gabinete: %s",
                pagina, len(registros), chave,
            )
            time.sleep(INTERVALO_REQ)

        self.logger.info(
            "Gabinete '%s' — %d processos extraídos.", chave, len(registros)
        )
        return registros


# ==============================================================================
# ETAPA 1 — EXTRAÇÃO E GRAVAÇÃO DOS PARQUET INTERMEDIÁRIOS
# ==============================================================================

def etapa1_extrair(
    cliente: ClienteDatajud,
    data_inicio: str,
    data_fim: str,
    output_dir: Path,
    logger: logging.Logger,
) -> Dict[str, Path]:
    """
    Extrai os registros brutos dos três gabinetes e grava um arquivo
    .parquet intermediário por gabinete em disco.

    O arquivo parquet permite:
      - Inspecionar os dados antes de processar
      - Retomar a Etapa 2 sem reextair se o processo for interrompido
      - Analisar os dados com pandas/jupyter independentemente

    Returns
    -------
    Dict[str, Path]
        Mapeamento chave_gabinete → caminho do parquet.
    """
    caminhos: Dict[str, Path] = {}

    for chave in GABINETES:
        caminho = output_dir / f"raw_{chave}.parquet"

        if caminho.exists():
            logger.info(
                "Parquet intermediário já existe: %s — pulando extração. "
                "Delete o arquivo para forçar nova extração.", caminho
            )
            caminhos[chave] = caminho
            continue

        registros = cliente.extrair_gabinete(
            chave, data_inicio, data_fim,
        )
        if not registros:
            logger.warning("Gabinete '%s' retornou 0 registros.", chave)
            continue

        df = pd.DataFrame(registros)
        # Serializa colunas JSON para gravação em parquet
        cols_json = [
            "classe", "sistema", "movimentos", "orgaoJulgador",
            "assuntos", "formato",
        ]
        for col in cols_json:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda v: json.dumps(v, ensure_ascii=False)
                    if not isinstance(v, str) else v
                )

        df.to_parquet(caminho, index=False)
        logger.info(
            "Parquet gravado: %s (%d processos, %d colunas)",
            caminho, len(df), len(df.columns),
        )
        caminhos[chave] = caminho

    return caminhos


# ==============================================================================
# ETAPA 2 — PROCESSAMENTO DOS PARQUET → XES + CSV + RELATÓRIO
# ==============================================================================

def _processar_movimentos(movimentos_raw: Any) -> List[Dict]:
    """
    Processa a lista de movimentos de um processo:
      - Deserializa se necessário
      - Aplica fix_mojibake
      - Extrai complementosTabelados
      - Resolve rótulo XES via Mapa TPU
    """
    if isinstance(movimentos_raw, str):
        try:
            movimentos_raw = json.loads(movimentos_raw)
        except json.JSONDecodeError:
            return []
    if not isinstance(movimentos_raw, list):
        return []

    resultado = []
    for mov in movimentos_raw:
        mov = fix_mojibake(mov)
        codigo = int(mov.get("codigo", 0))
        nome_tpu = mov.get("nome", "").strip()
        atividade = resolver_atividade(codigo, nome_tpu)
        data_hora = mov.get("dataHora", "")

        # Extrai complementosTabelados
        comps_raw = mov.get("complementosTabelados", [])
        complementos = []
        if isinstance(comps_raw, list):
            complementos = [
                {"codigo": c.get("codigo"), "valor": c.get("valor")}
                for c in comps_raw
                if isinstance(c, dict)
            ]

        resultado.append({
            "codigo_tpu": codigo,
            "nome_tpu": nome_tpu,
            "atividade_xes": atividade,
            "data_hora": data_hora,
            "complementos": complementos,
        })

    # Ordena cronologicamente
    resultado.sort(key=lambda m: m["data_hora"])
    return resultado


def _parsear_json_col(val: Any) -> Any:
    """Deserializa coluna armazenada como JSON string no parquet."""
    if isinstance(val, str):
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            return {}
    return val if val is not None else {}


def etapa2_processar(
    caminhos: Dict[str, Path],
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    """
    Lê os parquet intermediários e gera os artefatos finais:
      - Um arquivo XES por gabinete (formato PM4Py)
      - pm4jud_eventos.csv   (todos os eventos, todos os gabinetes)
      - pm4jud_processos.csv (metadados de processo, todos os gabinetes)
      - pm4jud_relatorio.json
    """
    todos_processos = []
    todos_eventos = []

    for chave, caminho in caminhos.items():
        gab = GABINETES[chave]
        logger.info("Processando parquet: %s", caminho)

        df = pd.read_parquet(caminho)

        # Deserializa colunas JSON
        for col in ["classe", "sistema", "movimentos", "orgaoJulgador",
                    "assuntos", "formato"]:
            if col in df.columns:
                df[col] = df[col].apply(_parsear_json_col)

        processos_gab = []
        for _, row in df.iterrows():
            numero = str(row.get("numeroProcesso", "")).strip()
            if not numero:
                continue

            classe_obj = _parsear_json_col(row.get("classe", {}))
            orgao_obj  = _parsear_json_col(row.get("orgaoJulgador", {}))
            assuntos = row.get("assuntos", [])
            if isinstance(assuntos, str):
                try:
                    assuntos = json.loads(assuntos)
                except Exception:
                    assuntos = []
            # Garante que cada item da lista é dict (pode vir como str aninhada)
            assuntos_limpos = []
            for a in (assuntos if isinstance(assuntos, list) else []):
                if isinstance(a, dict):
                    assuntos_limpos.append(a)
                elif isinstance(a, str):
                    try:
                        parsed = json.loads(a)
                        if isinstance(parsed, dict):
                            assuntos_limpos.append(parsed)
                    except Exception:
                        pass
            assuntos = assuntos_limpos

            codigo_classe = int(classe_obj.get("codigo", 0))
            nome_classe   = fix_mojibake(classe_obj.get("nome", ""))
            orgao_nome    = fix_mojibake(orgao_obj.get("nome", ""))
            prioritario   = codigo_classe in CLASSES_HC_PRIORITARIAS

            movimentos = _processar_movimentos(row.get("movimentos", []))

            proc = {
                "gabinete": chave,
                "ministro": gab["nome_completo"],
                "turma": gab["turma"],
                "secao": gab["secao"],
                "materia": gab["materia"],
                "numero_processo": numero,
                "codigo_classe": codigo_classe,
                "nome_classe": nome_classe,
                "data_ajuizamento": row.get("dataAjuizamento", ""),
                "orgao_julgador": orgao_nome,
                "grau": row.get("grau", ""),
                "nivel_sigilo": row.get("nivelSigilo", 0),
                "prioritario": prioritario,
                "classe_pm4jud": "prioritario" if prioritario else "regular",
                "n_movimentos": len(movimentos),
                "assuntos": "; ".join(
                    f"{a.get('codigo')}:{fix_mojibake(a.get('nome',''))}"
                    for a in assuntos
                ),
                "origem_dado": "DATAJUD",
                "movimentos": movimentos,
            }
            processos_gab.append(proc)
            todos_processos.append(proc)

            for mov in movimentos:
                todos_eventos.append({
                    "gabinete": chave,
                    "case_id": numero,
                    "activity": mov["atividade_xes"],
                    "timestamp": mov["data_hora"],
                    "resource": orgao_nome,
                    "codigo_tpu": mov["codigo_tpu"],
                    "nome_tpu": mov["nome_tpu"],
                    "codigo_classe": codigo_classe,
                    "nome_classe": nome_classe,
                    "prioritario": prioritario,
                    "classe_pm4jud": "prioritario" if prioritario else "regular",
                    "origem": "DATAJUD",
                })

        # Exporta XES por gabinete
        if PM4PY_DISPONIVEL and processos_gab:
            log_xes = _construir_log_xes(processos_gab, chave, gab)
            xes_path = output_dir / f"pm4jud_log_gab_{chave}.xes"
            xes_exporter.apply(log_xes, str(xes_path))
            logger.info("XES exportado: %s (%d traços)", xes_path, len(processos_gab))
        elif not PM4PY_DISPONIVEL:
            logger.warning("PM4Py não disponível — XES não gerado para '%s'.", chave)

    # CSV consolidado
    df_proc = pd.DataFrame([
        {k: v for k, v in p.items() if k != "movimentos"}
        for p in todos_processos
    ])
    df_proc.to_csv(output_dir / "pm4jud_processos.csv",
                   index=False, encoding="utf-8-sig")
    logger.info("CSV processos: %d linhas", len(df_proc))

    df_evt = pd.DataFrame(todos_eventos)
    df_evt.to_csv(output_dir / "pm4jud_eventos.csv",
                  index=False, encoding="utf-8-sig")
    logger.info("CSV eventos: %d linhas", len(df_evt))

    # Relatório JSON
    relatorio = _gerar_relatorio(todos_processos, todos_eventos)
    rel_path = output_dir / "pm4jud_relatorio.json"
    rel_path.write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Relatório JSON: %s", rel_path)
    return relatorio


def _construir_log_xes(
    processos: List[Dict],
    chave: str,
    gab: Dict,
) -> "EventLog":
    """Constrói EventLog PM4Py a partir dos processos processados."""
    log = EventLog()
    log.attributes["concept:name"] = f"PM4JUD — {gab['nome_completo']} ({gab['turma']})"
    log.attributes["pm4jud:gabinete"] = chave
    log.attributes["pm4jud:turma"]   = gab["turma"]
    log.attributes["pm4jud:secao"]   = gab["secao"]
    log.attributes["pm4jud:materia"] = gab["materia"]
    log.attributes["pm4jud:fase"] = "Fase 1 — DATAJUD público"
    log.attributes["pm4jud:data_extracao"] = datetime.now(timezone.utc).isoformat()
    log.attributes["xes.version"] = "2.0"

    for proc in processos:
        trace = Trace()
        # Atributos da trace — metadados do processo judicial
        # --------------------------------------------------------
        # Cada trace representa um processo e carrega atributos estáticos
        # (não mudam ao longo do tempo). Esses atributos são lidos pelos
        # programas downstream (P2, P3, P4, P5) para calibração e filtragem.
        #
        # pm4jud:codigo_classe   → código TPU da classe (ex.: "140" = HC)
        #                          Usado pelo COMPLEMENT para calibração Dirichlet
        #                          e por CLASSES_PRIORITARIAS para prioridade.
        # pm4jud:classe_tpu      → alias de codigo_classe (canônico PM4JUD)
        # pm4jud:materia         → matéria processual (Criminal, Cível, etc.)
        # pm4jud:prioritario     → bool — True se classe ou situação exige
        #                          julgamento prioritário (RISTJ art. 202)
        # pm4jud:assuntos        → lista de códigos de assunto TPU
        #                          (FamiliaHabeasCorpus: 1720,1013,1722,1064,15224)
        trace.attributes["concept:name"]       = proc["numero_processo"]
        trace.attributes["pm4jud:gabinete"]    = proc["gabinete"]
        trace.attributes["pm4jud:ministro"]    = proc["ministro"]
        trace.attributes["pm4jud:turma"]    = proc["turma"]
        trace.attributes["pm4jud:secao"]    = proc.get("secao", "")
        trace.attributes["pm4jud:materia"]  = proc.get("materia", "")
        trace.attributes["pm4jud:codigo_classe"] = proc["codigo_classe"]
        trace.attributes["pm4jud:classe_tpu"]    = proc["codigo_classe"]  # alias canônico PM4JUD
        trace.attributes["pm4jud:nome_classe"] = proc["nome_classe"]
        trace.attributes["pm4jud:prioritario"] = proc["prioritario"]
        trace.attributes["pm4jud:classe_pm4jud"] = proc["classe_pm4jud"]
        trace.attributes["pm4jud:assuntos"]    = proc["assuntos"]
        trace.attributes["pm4jud:origem"]      = "DATAJUD"

        for mov in proc["movimentos"]:
            evento = Event()
            evento["concept:name"]        = mov["atividade_xes"]
            evento["pm4jud:codigo_tpu"]   = mov["codigo_tpu"]
            evento["pm4jud:nome_tpu"]     = mov["nome_tpu"]
            evento["org:resource"]        = proc["orgao_julgador"]
            evento["pm4jud:origem_resource"] = "DATAJUD_proxy"

            try:
                ts = mov["data_hora"].strip()
                for fmt in [
                    "%Y-%m-%dT%H:%M:%S.%fZ",
                    "%Y-%m-%dT%H:%M:%SZ",
                    "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%d",
                ]:
                    try:
                        dt = datetime.strptime(ts, fmt).replace(
                            tzinfo=timezone.utc
                        )
                        evento["time:timestamp"] = dt
                        break
                    except ValueError:
                        continue
            except Exception:
                evento["time:timestamp"] = None

            trace.append(evento)
        log.append(trace)
    return log


def _gerar_relatorio(
    processos: List[Dict],
    eventos: List[Dict],
) -> Dict[str, Any]:
    """Gera dicionário de estatísticas para o relatório JSON."""
    por_gabinete: Dict[str, int] = {}
    por_classe: Dict[str, int] = {}
    por_prioridade = {"prioritario": 0, "regular": 0}

    for p in processos:
        g = p["gabinete"]
        nc = p["nome_classe"]
        por_gabinete[g] = por_gabinete.get(g, 0) + 1
        por_classe[nc]  = por_classe.get(nc, 0) + 1
        por_prioridade[p["classe_pm4jud"]] += 1

    por_atividade: Dict[str, int] = {}
    for e in eventos:
        a = e["activity"]
        por_atividade[a] = por_atividade.get(a, 0) + 1

    total_proc = len(processos)
    total_evt  = len(eventos)
    media_evt  = round(total_evt / total_proc, 2) if total_proc else 0

    return {
        "pm4jud_etl_versao": "2.0",
        "data_extracao": datetime.now(timezone.utc).isoformat(),
        "tribunal": "STJ",
        "secao": "3 Seções do STJ (1ª Pública + 2ª Privada + 3ª Criminal)",
        "filtros": {
            "publicacao_tpu": CODIGOS_PUBLICACAO,
            "descricao": "Apenas processos com publicação (traço completo) — censo 2024",
        },
        "totais": {
            "processos": total_proc,
            "eventos": total_evt,
            "media_eventos_por_processo": media_evt,
        },
        "por_gabinete": dict(
            sorted(por_gabinete.items(), key=lambda x: x[1], reverse=True)
        ),
        "por_classe": dict(
            sorted(por_classe.items(), key=lambda x: x[1], reverse=True)[:20]
        ),
        "por_prioridade": por_prioridade,
        "top_20_atividades": dict(
            sorted(por_atividade.items(), key=lambda x: x[1], reverse=True)[:20]
        ),
    }


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PM4JUD-ETL v2.0 — Extração DATAJUD por gabinete criminal STJ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-inicio", default=DATA_INICIO_PADRAO,
        help=f"Data inicial YYYY-MM-DD (padrão: {DATA_INICIO_PADRAO})",
    )
    parser.add_argument(
        "--data-fim", default=DATA_FIM_PADRAO,
        help=f"Data final YYYY-MM-DD (padrão: {DATA_FIM_PADRAO})",
    )
    parser.add_argument(
        "--output-dir", default=str(OUTPUT_DIR_PADRAO),
        help=f"Diretório de saída (padrão: {OUTPUT_DIR_PADRAO})",
    )

    parser.add_argument(
        "--ontologia",
        type=Path,
        default=None,
        help=(
            "Diretório da ontologia PM4JUD (contém PM4JUD_Movimentos.owl). "
            "Padrão: ../ontologia/ relativo ao ETL."
        ),
    )
    parser.add_argument(
        "--apenas-etapa", choices=["1", "2"],
        help="Executa apenas a Etapa 1 (extração) ou Etapa 2 (processamento)",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="Chave API DPJ/CNJ. Se omitida, lê DATAJUD_API_KEY do .env",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configurar_logging(output_dir)

    # Resolve chave API
    api_key = (args.api_key or os.environ.get("DATAJUD_API_KEY", "")).strip()
    if not api_key:
        logger.error(
            "Chave API não encontrada. Use --api-key ou defina "
            "DATAJUD_API_KEY no .env"
        )
        sys.exit(1)

    logger.info("=" * 70)
    # Carrega Ontologia PM4JUD — camada semântica transversal ao pipeline
    global MAPA_TPU
    from pm4jud_ontologia import carregar_ontologia
    ontologia_dir = args.ontologia or ONTOLOGIA_DIR_PADRAO
    ont = carregar_ontologia(ontologia_dir, modulos=[3, 5])
    MAPA_TPU = ont.mapa_tpu()
    if not MAPA_TPU:
        logger.warning(
            "MAPA_TPU vazio — nomes do DATAJUD usados como fallback."
        )
    # Propaga para funções que dependem de MAPA_TPU carregado da ontologia
    _ONTOLOGIA_INSTANCIA = ont  # noqa: disponível para módulos downstream

    logger.info("PM4JUD-ETL v2.0 — Início")
    logger.info("Período   : %s → %s", args.data_inicio, args.data_fim)
    logger.info("Volume    : acervo completo de 2024 (sem limite)")
    logger.info("Filtro    : publicação TPU %s", CODIGOS_PUBLICACAO)
    logger.info("Saídas em : %s", output_dir.resolve())
    logger.info("=" * 70)

    cliente = ClienteDatajud(api_key=api_key)

    # ETAPA 1
    if args.apenas_etapa != "2":
        logger.info("── ETAPA 1: Extração e gravação dos parquet ──────────────────")
        caminhos = etapa1_extrair(
            cliente, args.data_inicio, args.data_fim,
            output_dir, logger,
        )
    else:
        # Localiza parquet existentes
        caminhos = {
            chave: output_dir / f"raw_{chave}.parquet"
            for chave in GABINETES
            if (output_dir / f"raw_{chave}.parquet").exists()
        }
        if not caminhos:
            logger.error(
                "Nenhum parquet encontrado em %s. Execute a Etapa 1 primeiro.",
                output_dir,
            )
            sys.exit(1)

    # ETAPA 2
    if args.apenas_etapa != "1":
        logger.info("── ETAPA 2: Processamento e exportação ───────────────────────")
        relatorio = etapa2_processar(caminhos, output_dir, logger)
        logger.info("=" * 70)
        logger.info("PM4JUD-ETL — Pipeline concluído.")
        logger.info(
            "  Processos totais : %d", relatorio["totais"]["processos"]
        )
        logger.info(
            "  Eventos totais   : %d", relatorio["totais"]["eventos"]
        )
        for gab, n in relatorio["por_gabinete"].items():
            logger.info(
                "  %-12s : %d processos", gab, n
            )
        logger.info("=" * 70)
    else:
        logger.info("Etapa 1 concluída. Parquet gravados em %s", output_dir)


if __name__ == "__main__":
    main()
