#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
PM4JUD-VOCAB  v1.0
================================================================================

Dissertação de Mestrado — PPGIa/PUCPR
Título: PM4JUD — Otimização Multiobjetivo com Mineração de Processos e
        Simulação no Contexto do Fluxo Processual em Gabinetes de Magistrado
Autor:  Luiz Claudio Soares de Almeida
Orient: Prof. Dr. Edson Emilio Scalabrin
Ano:    2026

Descrição
---------
Módulo de vocabulário canônico compartilhado entre todos os programas do
pipeline PM4JUD. Centraliza:

  1. CANONICAL_MAP — mapeamento de rótulos brutos TPU → rótulos canônicos XES.
     Fonte autoritativa: PM4JUD_Movimentos.owl (Módulo 4 da Ontologia PM4JUD).
     Usado pelo P1 (ETL) e pelo P3 (COMPLEMENT) via importação direta.

  2. _limpar_tipo_movimento() — normaliza templates #{...} e #(...) do
     atributo stj:tipoMovimento extraído da ontologia.

  3. CODIGOS_COLEGIADA — conjunto de códigos TPU de movimentos que indicam
     tramitação em sessão colegiada (julgamento em turma ou seção).

  4. CODIGOS_CLASSE_PRIORITARIA — conjunto de códigos TPU de classes processuais
     com prioridade regimental de julgamento (RISTJ art. 202).

  5. carregar_rotulos_colegiada() — consulta SPARQL sobre PM4JUD_Movimentos.owl
     e retorna os rótulos canônicos XES dos movimentos colegiados.
     Usado pelo P3 (COMPLEMENT) para substituir a constante hardcoded
     ROTULOS_COLEGIADA.

  6. carregar_classes_prioritarias() — consulta SPARQL sobre PM4JUD_Classes.owl
     e retorna os códigos de string das classes processuais prioritárias.
     Usado pelo P3 (COMPLEMENT) para substituir CLASSES_PRIORITARIAS.

Princípio arquitetural
----------------------
A Ontologia PM4JUD é a FONTE AUTORITATIVA de classificação taxonômica dos
movimentos e classes processuais. Os rótulos que aparecem como concept:name
nos logs XES devem corresponder a indivíduos da ontologia. Este módulo garante
que essa correspondência seja mantida em tempo de execução, carregando os
rótulos diretamente da ontologia via rdflib + SPARQL, em vez de hardcoded.

Fallback gracioso
-----------------
Se a ontologia não estiver disponível (rdflib ausente, arquivo não encontrado,
erro de parse), todas as funções retornam os conjuntos padronizados derivados
do CANONICAL_MAP — garantindo que o pipeline não quebre.

Referências
-----------
LENZERINI, M. et al. Metamodeling and Punning in OWL 2. Artificial
  Intelligence, v. 292, p. 103432, 2021.
  DOI 10.1016/j.artint.2020.103432.

STJ/CNJ. Tabela de Movimentos Processuais Unificada (TPU). Versão
  09/04/2026. Brasília: CNJ, 2026.

STJ. Regimento Interno do Superior Tribunal de Justiça. Brasília: STJ, 2024.

Repositório: https://github.com/luizcsalmeida/pm4jud/tree/main/etl
================================================================================
"""

import logging
import re
from pathlib import Path
from typing import Dict, Set

try:
    from rdflib import Graph
    RDFLIB_DISPONIVEL = True
except ImportError:
    RDFLIB_DISPONIVEL = False

# ==============================================================================
# PREFIXOS SPARQL
# ==============================================================================

PREFIX_STJ  = "http://www.stj.jus.br/mni/intercomunicacao#"
PREFIX_RDF  = "http://www.w3.org/2000/01/rdf-schema#"

SPARQL_PREFIXES = """
    PREFIX stj:  <http://www.stj.jus.br/mni/intercomunicacao#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX owl:  <http://www.w3.org/2002/07/owl#>
"""


# ==============================================================================
# CANONICAL_MAP — Vocabulário Canônico PM4JUD
# ==============================================================================
# Mapeamento de rótulos brutos (stj:tipoMovimento limpo) → rótulos canônicos
# usados como concept:name nos logs XES do pipeline PM4JUD.
#
# Princípio: a ontologia PM4JUD_Movimentos.owl é a fonte taxonômica primária.
# Este mapa resolve duas necessidades complementares:
#   (a) padroniza variantes textuais do mesmo movimento (template cleanup);
#   (b) produz rótulos mais curtos/concisos para mineração de processos,
#       reduzindo a fragmentação do alfabeto de atividades.
#
# Técnica de punning (Lenzerini et al., 2021): cada movimento TPU é declarado
# simultaneamente como owl:Class (taxonomia) e owl:NamedIndividual (instância
# referenciável no log XES). O CANONICAL_MAP resolve o mapeamento de texto
# livre → IRI de indivíduo.
#
# Atualização: v2.1 — inclui movimentos identificados em processos reais
# HC 881458/SP e HC 881499/MG (STJ, mai/2026).
# ==============================================================================

CANONICAL_MAP: Dict[str, str] = {
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
    # Liminar (TPU 339, 792, 892, 348, 12207)
    "Ratificada a liminar":                                         "Liminar ratificada",
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
    # Recebimento / Julgamento
    "Recebidos os autos":                                           "Recebimento",
    "Julgamento":                                                   "Julgado",
    # Redistribuição — variantes residuais após limpeza de template
    "Redistribuído por em razão":                                   "Redistribuído",
    "Redistribuído por em razão de":                                "Redistribuído",
    # Expedição — residual
    "Expedição de":                                                 "Mandado expedido",
    # Trânsito em julgado — variantes (TPU 848)
    "Transitado em Julgado":                                        "Trânsito em julgado",
    "Transitado em julgado":                                        "Trânsito em julgado",
    # Arquivamento definitivo — variantes (TPU 246)
    "Determinado o arquivamento definitivo":                        "Arquivado definitivamente",
    "Arquivado Definitivamente":                                    "Arquivado definitivamente",

    # ── ADICIONADOS v2.1 ── Processos reais HC 881458/SP e HC 881499/MG ───────
    # Strings após _limpar_tipo_movimento() dos tipoMovimento da ontologia:
    # Validadas via SPARQL no Protégé (mai/2026)
    "Conhecido o recurso e não-provido":                        "Recurso conhecido e não provido",
    "Não conhecido o Habeas Corpus":                            "HC não conhecido",
    "Não conhecido o Habeas Corpus. Concedido o Habeas Corpus de ofício":
                                                                "HC concedido de ofício",
    "Denegado o Habeas Corpus":                                 "HC denegado",
    "Concedido em parte o Habeas Corpus":                       "HC concedido parcialmente",
    # Conclusos (TPU 51)
    "Conclusos para decisão":                                       "Conclusão para decisão",
    "Conclusos para julgamento":                                    "Conclusão para julgamento",
    # Liminar não concedida — variante lowercase (TPU 792)
    "Não concedida a medida liminar":                               "Liminar indeferida",
    "Não concedida a Medida Liminar":                               "Liminar indeferida",
    # Despacho de mero expediente (TPU 11010)
    "Proferido despacho de mero expediente":                        "Despacho de mero expediente",
    "Proferido despacho":                                           "Despacho de mero expediente",
    # Inclusão em pauta / mesa
    # TPU 417 no DATAJUD/STJ (não 3002) — rdfs:label = "Inclusão em pauta"
    # tipoMovimento: "Incluído em pauta para #{data_hora} #{local}."
    # após _limpar_tipo_movimento (2 passos): "Incluído em pauta" → canônico abaixo
    "Incluído em pauta":                                            "Inclusão em pauta",
    "Inclusão em mesa para julgamento":                             "Inclusão em pauta",
    "Incluído em mesa para julgamento":                             "Inclusão em pauta",
    # Compatibilidade retroativa: XES gerado antes do fix do _limpar (trailing "para")
    "Incluído em pauta para":                                       "Inclusão em pauta",
    # Resultado colegiado (TPU 239)
    "Conhecido o recurso e não-provido":                            "Recurso conhecido e não provido",
    "Conhecido o recurso":                                          "Recurso conhecido",
    "Não conhecido o recurso":                                      "Recurso não conhecido",
    "Provido o recurso":                                            "Recurso provido",
    "Não provido o recurso":                                        "Recurso não provido",
    # Proclamação Final de Julgamento (TPU 3001)
    "Proclamação Final de Julgamento":                              "Proclamação de julgamento",
    "Proclamação de Julgamento":                                    "Proclamação de julgamento",
}


# ==============================================================================
# NORMALIZAÇÃO DE TEMPLATES
# ==============================================================================

def _limpar_tipo_movimento(tipo: str) -> str:
    """
    Remove templates #{...} e #(...) do atributo stj:tipoMovimento da ontologia.

    Contexto
    --------
    Os movimentos processuais no SGT/STJ usam templates para descrever ações
    com partes variáveis. Por exemplo:
      "Concedido o Habeas Corpus a #{nome_da_parte} (#{papel})"
    O template #{nome_da_parte} é substituído pelo nome real do litigante em
    cada processo, mas a ontologia armazena o template completo — não o valor.

    Esta função extrai a parte invariante (o "tipo" semântico do movimento)
    e a normaliza para uso como concept:name no XES:

    Exemplos de transformação
    -------------------------
    "Concedido o Habeas Corpus a #{nome_da_parte}"  → "Concedido o Habeas Corpus"
    "Publicado #{ato_publicado} em #{data}."         → "Publicado"
    "Disponibilizado no DJ Eletrônico em #(data)"   → "Disponibilizado no DJ Eletrônico"
    "Inclusão em mesa para julgamento - #{orgao}"   → "Inclusão em mesa para julgamento"

    Passos de limpeza (em ordem)
    ----------------------------
    1. Remove #{...} — templates de variável nomeada.
    2. Remove #(...) — templates de variável posicional (formato alternativo STJ).
    3. Remove preposições/artigos isolados no final — resíduos do template.
       Ex.: "Publicado em" → "Publicado" (o "em" ficou sem argumento).
    4. Normaliza múltiplos espaços para espaço único.
    5. Remove pontuação final (ponto, vírgula, ponto-e-vírgula).

    Nota: após a limpeza, o CANONICAL_MAP é aplicado para resolver variantes
    textuais remanescentes — as duas etapas são complementares.

    Parameters
    ----------
    tipo : str
        Valor bruto do atributo stj:tipoMovimento extraído da ontologia.

    Returns
    -------
    str
        Rótulo limpo, pronto para lookup no CANONICAL_MAP e uso como
        concept:name no log XES.
    """
    limpo = re.sub(r"#\{[^}]+\}", "", tipo)
    limpo = re.sub(r"#\([^)]+\)", "", limpo)
    # Remove preposição/artigo isolado no MEIO da frase antes de conjunção "e".
    # Ocorre quando o template #{...} estava entre uma preposição e uma conjunção.
    # Exemplo: "recurso de #{nome} e não-provido" → remove "de" → "recurso e não-provido"
    limpo = re.sub(
        r"\b(a|ao|aos|de|do|da|dos|das|em|no|na|nos|nas|para|por|pelo|pela|o|os)\s+e\s+",
        " e ", limpo, flags=re.IGNORECASE,
    )
    # Remove preposição/artigo isolado no FINAL da frase (resíduo de template terminal).
    limpo = re.sub(
        r"\s+(a|ao|aos|de|do|da|dos|das|em|no|na|nos|nas|para|por|pelo|pela|o|os)\s*$",
        "", limpo.strip(), flags=re.IGNORECASE,
    )
    limpo = re.sub(r"\s+", " ", limpo)
    limpo = re.sub(r"[\s.,;:]+$", "", limpo.strip())
    # Segunda passagem: remove preposição que ficou exposta após remoção de pontuação.
    # Caso típico: "Incluído em pauta para #{data} #{local}."
    #   → remove templates → "Incluído em pauta para ."
    #   → remove "." → "Incluído em pauta para"   ← o "para" só fica visível agora
    #   → segunda passagem remove o "para" → "Incluído em pauta"
    limpo = re.sub(
        r"\s+(a|ao|aos|de|do|da|dos|das|em|no|na|nos|nas|para|por|pelo|pela|o|os)\s*$",
        "", limpo.strip(), flags=re.IGNORECASE,
    )
    limpo = limpo.strip()
    return limpo if limpo else tipo


# ==============================================================================
# CONSTANTES DERIVADAS DA ONTOLOGIA
# ==============================================================================

# Códigos TPU de movimentos que indicam tramitação em sessão colegiada.
# Fonte: PM4JUD_Movimentos.owl (Módulo 4) — indivíduos cujo rótulo canônico
# aparece em sessões de turma ou seção do STJ.
# Referência: RISTJ arts. 94, 95, 177, II, 202.
CODIGOS_COLEGIADA: Set[int] = {
    # ── Confirmados no corpus DATAJUD 2024 (conciliacao_tpu.xlsx mai/2026) ──
    417,    # Inclusão em pauta — código real DATAJUD/STJ (não 3002)
    239,    # Recurso conhecido e não provido
    235,    # Não conhecido o recurso
    237,    # Recurso provido
    238,    # Recurso provido em parte
    242,    # Conhecido em parte e não provido
    447,    # HC denegado
    451,    # HC concedido parcialmente
    12458,  # HC não conhecido
    12475,  # HC concedido de ofício
    14093,  # Voto do relator proferido
    12204,  # Deliberado em Sessão - Pedido de Vista
    12309,  # Retirada de pauta
    12106,  # Adiamento do julgamento para primeira sessão seguinte
    # ── Não confirmados no corpus 2024 (ausentes do DATAJUD STJ) ───────────
    # 3001: Proclamação de julgamento — não usado pelo STJ/DATAJUD (redundante com resultados)
    # 3002: Inclusão em mesa — código CNJ genérico; STJ usa 417
    # 243:  Origem incerta — removido até confirmação no SGT/CNJ
}

# Códigos TPU de classes processuais com prioridade regimental.
# Fonte: PM4JUD_Classes.owl (Módulo 3) — RISTJ art. 202.
CODIGOS_CLASSE_PRIORITARIA: Set[int] = {
    140,    # Habeas Corpus — réu preso (prioridade absoluta)
    854,    # Recurso em Habeas Corpus
    143,    # Habeas Data
    1200,   # Mandado de Segurança (prioridade parcial)
}


# ==============================================================================
# LOADERS SPARQL — Ontologia → constantes em runtime
# ==============================================================================

def carregar_rotulos_colegiada(
    ontologia_dir: Path,
    logger: logging.Logger | None = None,
) -> Set[str]:
    """
    Carrega da Ontologia PM4JUD os rótulos canônicos dos movimentos colegiados.

    Motivação arquitetural
    ----------------------
    O conjunto ROTULOS_COLEGIADA é usado pelo COMPLEMENT (P3) para decidir
    qual documento decisório gerar: RELATÓRIO E VOTO (colegiado) ou
    DESPACHO/DECISÃO (monocrático). Se esse conjunto for hardcoded, qualquer
    atualização da tabela TPU/CNJ exigiria recompilar o código.

    Ao carregar da ontologia em runtime, garantimos que a detecção do caminho
    decisório seja sempre consistente com a versão atual do PM4JUD_Movimentos.owl.

    Estratégia de consulta
    ----------------------
    Consulta PM4JUD_Movimentos.owl via rdflib com SPARQL, usando VALUES para
    filtrar pelos CODIGOS_COLEGIADA (constante arquitetural — representa os
    códigos TPU que semanticamente indicam sessão colegiada).

    Para cada código, recupera:
      1. stj:tipoMovimento → aplica _limpar_tipo_movimento()
      2. rdfs:label (fallback se tipoMovimento ausente)

    O resultado é mapeado pelo CANONICAL_MAP para obter o concept:name exato
    que aparece nos logs XES (gerado pelo ETL com o mesmo mapa).

    Fallback gracioso
    -----------------
    Se rdflib não estiver instalado, o arquivo OWL não for encontrado, ou
    a consulta retornar vazio: retorna _rotulos_colegiada_padrao() — um
    conjunto hardcoded derivado do CANONICAL_MAP para os códigos conhecidos.
    O pipeline não falha; apenas loga um WARNING.
    

    O CANONICAL_MAP é aplicado sobre os rótulos brutos da ontologia para
    garantir que o conjunto retornado corresponda exatamente aos concept:name
    presentes nos logs XES gerados pelo PM4JUD-ETL (P1).

    Arquitetura
    -----------
    A Ontologia PM4JUD é a fonte autoritativa dos movimentos. Este método
    substitui a constante hardcoded ROTULOS_COLEGIADA no PM4JUD-COMPLEMENT (P3),
    garantindo que qualquer atualização na ontologia se propague automaticamente.

    Parameters
    ----------
    ontologia_dir : Path
        Diretório com PM4JUD_Movimentos.owl e módulos MNI importados.
    logger : logging.Logger, opcional
        Logger para diagnóstico. Se None, usa o logger raiz.

    Returns
    -------
    Set[str]
        Conjunto de rótulos canônicos XES. Em caso de falha, retorna o
        conjunto padronizado derivado do CANONICAL_MAP (fallback gracioso).
    """
    log = logger or logging.getLogger("PM4JUD-VOCAB")

    if not RDFLIB_DISPONIVEL:
        log.warning("rdflib não disponível — usando ROTULOS_COLEGIADA padronizados.")
        return _rotulos_colegiada_padrao()

    owl_path = ontologia_dir / "PM4JUD_Movimentos.owl"
    if not owl_path.exists():
        log.warning("Ontologia não encontrada: %s — usando fallback.", owl_path)
        return _rotulos_colegiada_padrao()

    try:
        g = Graph()
        g.parse(str(owl_path), format="xml")

        # IRI completo obrigatório para typed literals em VALUES com rdflib.
        # PREFIX xsd: declarado no SPARQL_PREFIXES não é resolvido dentro de
        # VALUES pelo motor SPARQL do rdflib — usar IRI explícito garante o
        # match com "3002"^^<http://...#integer> armazenado na ontologia.
        xsd_int = "http://www.w3.org/2001/XMLSchema#integer"
        valores = " ".join(f'"{c}"^^<{xsd_int}>' for c in CODIGOS_COLEGIADA)
        sparql = f"""
            {SPARQL_PREFIXES}
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

            SELECT DISTINCT ?tipo ?label WHERE {{
                VALUES ?cod {{ {valores} }}
                ?mov stj:codigoMovimentoTPU ?cod .
                OPTIONAL {{ ?mov stj:tipoMovimento ?tipo }}
                OPTIONAL {{ ?mov rdfs:label ?label . FILTER(lang(?label) = "pt") }}
            }}
        """

        rotulos: Set[str] = set()
        for row in g.query(sparql):
            raw = ""
            if row.tipo and str(row.tipo).strip():
                raw = _limpar_tipo_movimento(str(row.tipo))
            elif row.label and str(row.label).strip():
                raw = str(row.label).strip()
            if raw:
                canonico = CANONICAL_MAP.get(raw, raw)
                rotulos.add(canonico)

        if rotulos:
            # Suplementa com os 3 códigos ausentes da ontologia (TPU 3001, 3002, 243).
            # Esses movimentos NÃO estão em PM4JUD_Movimentos.owl nesta versão —
            # são os indicadores pré-sessão mais importantes (Inclusão em pauta,
            # Proclamação de julgamento, Recurso provido). Devem ser adicionados
            # ao OWL em próxima revisão da ontologia.
            # TODO: adicionar Movimento_3001, Movimento_3002, Movimento_243 ao OWL.
            suplemento = {
                "Inclusão em pauta",         # TPU 3002 — ausente do OWL
                "Proclamação de julgamento", # TPU 3001 — ausente do OWL
                "Recurso provido",           # TPU 243  — ausente do OWL
            }
            rotulos_finais = rotulos | suplemento
            log.info(
                "ROTULOS_COLEGIADA carregados da ontologia: %d rótulos (%s) "
                "+ %d suplementados (TPU 3001/3002/243 ausentes do OWL)",
                len(rotulos), owl_path.name, len(suplemento),
            )
            return rotulos_finais

        log.warning("Consulta SPARQL retornou vazio — usando fallback.")
        return _rotulos_colegiada_padrao()

    except Exception as exc:
        log.warning("Erro ao carregar ontologia (%s) — usando fallback.", exc)
        return _rotulos_colegiada_padrao()


def carregar_classes_prioritarias(
    ontologia_dir: Path,
    logger: logging.Logger | None = None,
) -> Set[str]:
    """
    Consulta PM4JUD_Classes.owl via rdflib e retorna os códigos de string
    das classes processuais com prioridade regimental.

    O conjunto retornado é compatível com o atributo pm4jud:codigo_classe
    gravado nas traces do XES pelo PM4JUD-ETL (P1): sempre strings de inteiros
    (ex.: "140", "854"), não inteiros puros.

    Parameters
    ----------
    ontologia_dir : Path
        Diretório com PM4JUD_Classes.owl.
    logger : logging.Logger, opcional

    Returns
    -------
    Set[str]
        Códigos de classe como strings. Fallback = hardcoded padronizado.
    """
    log = logger or logging.getLogger("PM4JUD-VOCAB")

    if not RDFLIB_DISPONIVEL:
        log.warning("rdflib não disponível — usando CLASSES_PRIORITARIAS padronizadas.")
        return _classes_prioritarias_padrao()

    owl_path = ontologia_dir / "PM4JUD_Classes.owl"
    if not owl_path.exists():
        log.warning("Ontologia não encontrada: %s — usando fallback.", owl_path)
        return _classes_prioritarias_padrao()

    try:
        g = Graph()
        g.parse(str(owl_path), format="xml")

        xsd_int = "http://www.w3.org/2001/XMLSchema#integer"
        valores = " ".join(f'"{c}"^^<{xsd_int}>' for c in CODIGOS_CLASSE_PRIORITARIA)
        sparql = f"""
            {SPARQL_PREFIXES}
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

            SELECT DISTINCT ?cod WHERE {{
                VALUES ?cod {{ {valores} }}
                ?cls stj:codigoClasseTPU ?cod .
            }}
        """

        codigos: Set[str] = set()
        for row in g.query(sparql):
            try:
                codigos.add(str(int(str(row.cod))))
            except (ValueError, AttributeError):
                continue

        if codigos:
            log.info(
                "CLASSES_PRIORITARIAS carregadas da ontologia: %s (%s)",
                sorted(codigos), owl_path.name,
            )
            return codigos

        log.warning("Consulta SPARQL retornou vazio — usando fallback.")
        return _classes_prioritarias_padrao()

    except Exception as exc:
        log.warning("Erro ao carregar ontologia (%s) — usando fallback.", exc)
        return _classes_prioritarias_padrao()


# ==============================================================================
# FALLBACKS PADRONIZADOS
# ==============================================================================

def _rotulos_colegiada_padrao() -> Set[str]:
    """
    Conjunto padronizado de rótulos canônicos XES para movimentos colegiados.
    Usado quando a ontologia não está disponível (fallback completo).
    Inclui os 6 mapeados via SPARQL + os 3 ausentes do OWL (3001, 3002, 243).
    """
    return {
        # Indicadores pré-sessão (TPU 3001, 3002 — ausentes do OWL mai/2026)
        "Inclusão em pauta",                # TPU 3002
        "Proclamação de julgamento",        # TPU 3001
        # Resultados colegiados (encontrados via SPARQL na ontologia)
        "Recurso conhecido e não provido",  # TPU 239
        "Recurso provido",                  # TPU 243 — ausente do OWL mai/2026
        "Recurso não provido",
        "Recurso não conhecido",
        "Voto do relator",                  # TPU 14093
        "HC concedido parcialmente",        # TPU 451
        "HC não conhecido",                 # TPU 12458
        "HC concedido de ofício",           # TPU 12475
        "HC denegado",                      # TPU 447
    }


def _classes_prioritarias_padrao() -> Set[str]:
    """
    Conjunto padronizado de códigos de classe processual prioritária.
    Derivado de CODIGOS_CLASSE_PRIORITARIA — fallback quando OWL indisponível.
    """
    return {str(c) for c in CODIGOS_CLASSE_PRIORITARIA}


# ==============================================================================
# K_POR_GABINETE — Limiar de frequência calibrado empiricamente por gabinete
# ==============================================================================
# Usado pelo P2 REFINE_1 (movimentos TPU) e P4 REFINE_2 (log completo).
# Valores derivados da maximização do MF1 no corpus DATAJUD 2024
# (32.031 processos — Rey=11.395, Pal=10.148, Sch=10.488).
# Fonte: pm4jud_diagnostico_tpu.py
K_POR_GABINETE: Dict[str, float] = {
    "reynaldo": 0.20,   # MF1=92,1%
    "palheiro": 0.30,   # MF1=77,5%
    "schietti": 0.25,   # MF1=81,9%
}


# ==============================================================================
# CANONICO_INTERNO — Canonicalização de rótulos de atividades SAGWeb
# ==============================================================================
# Mapeamento de rótulos legados ou variantes → rótulos canônicos v4.
# Usado pelo P3 COMPLEMENT (geração) e P4 REFINE_2 (Perfil 1 — canonicalização).
CANONICO_INTERNO: Dict[str, str] = {
    "Escaninho: Em analise":                         "Em analise",
    "Escaninho: Recebido":                           "Recebido pelo assessor",
    "Escaninho: Aguardando julgamento":              "Aguardando sessao",
    "Assinatura: Ministro":                          "Assinatura de documento pelo ministro",
    "Deslocamento: Para turma":                      "Deslocamento: Gabinete para Turma",
    # Legado v3 → canônico v4
    "Criacao de documento: Relatorio conclusivo":    "Criacao de documento: RELATORIO E VOTO",
    "Criacao de documento: Minuta de voto":          "Criacao de documento: RELATORIO E VOTO",
    "Alteracao de documento: Relatorio conclusivo":  "Alteracao de documento: RELATORIO E VOTO",
    "Alteracao de documento: Minuta de voto":        "Alteracao de documento: RELATORIO E VOTO",
}


# ==============================================================================
# NAO_RELABELAR — Atividades que NÃO recebem sufixo _N no Perfil 3
# ==============================================================================
# Eventos terminais ou únicos por trace que não devem ser relabelados mesmo
# quando ocorrem mais de uma vez (o que indica múltiplos julgamentos —
# situação tratada pelo modelo de processo, não pelo pré-processamento).
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


# ==============================================================================
# UTILITÁRIO — Relatórios com merge por gabinete
# ==============================================================================
# Todos os programas P1–P9 usam esta função para garantir que execuções
# individuais (um gabinete por vez) NUNCA sobrescrevam entradas anteriores.
# Cada execução atualiza apenas os gabinetes processados, preservando os demais.

import json as _json
from pathlib import Path as _Path
from datetime import datetime as _datetime, timezone as _timezone
from typing import Any as _Any, Dict as _Dict, List as _List


def salvar_relatorio_com_merge(
    fpath: _Path,
    programa: str,
    versao: str,
    novos_resultados: _List[_Dict],
    chave_gabinete: str = "gabinete",
    campos_consolidacao: _Dict[str, str] = None,
    extras: _Dict[str, _Any] = None,
) -> None:
    """
    Salva um relatório JSON com merge por gabinete.

    Carrega o arquivo existente, substitui apenas as entradas dos gabinetes
    recém-processados e reconstrói a seção de consolidação.

    Parameters
    ----------
    fpath : Path
        Caminho do arquivo JSON de saída.
    programa : str
        Nome do programa (ex: "PM4JUD-PM").
    versao : str
        Versão do programa.
    novos_resultados : list[dict]
        Lista de resultados desta execução. Cada dict deve ter a chave
        identificada por chave_gabinete.
    chave_gabinete : str
        Nome do campo que identifica o gabinete (padrão: "gabinete").
    campos_consolidacao : dict, optional
        Mapeamento {campo_soma: label} para totais na consolidação.
        Ex: {"traces": "total_traces", "eventos_total": "total_eventos"}
    extras : dict, optional
        Campos adicionais a incluir no payload raiz.
    """
    # Carrega estado anterior
    gabinetes_dict: _Dict[str, _Dict] = {}
    if fpath.exists():
        try:
            anterior = _json.loads(fpath.read_text(encoding="utf-8"))
            for g in anterior.get("gabinetes_lista", []):
                if chave_gabinete in g:
                    gabinetes_dict[g[chave_gabinete]] = g
        except Exception:
            pass

    # Merge: sobrescreve apenas os recém-processados
    for r in novos_resultados:
        r["timestamp_execucao"] = _datetime.now(_timezone.utc).isoformat()
        gabinetes_dict[r[chave_gabinete]] = r

    todos = list(gabinetes_dict.values())

    # Consolidação automática por soma
    consolidacao: _Dict[str, _Any] = {
        "gabinetes_presentes": sorted(gabinetes_dict.keys()),
        "gabinetes_faltando":  [g for g in ["reynaldo", "palheiro", "schietti"]
                                if g not in gabinetes_dict],
        "completo":            len(gabinetes_dict) == 3,
    }
    for campo, label in (campos_consolidacao or {}).items():
        consolidacao[label] = sum(
            v.get(campo, 0) for v in todos
            if isinstance(v.get(campo), (int, float))
        )

    payload: _Dict[str, _Any] = {
        "programa":           programa,
        "versao":             versao,
        "ultima_atualizacao": _datetime.now(_timezone.utc).isoformat(),
        "consolidacao":       consolidacao,
        "gabinetes_lista":    todos,
        "gabinetes_dict":     gabinetes_dict,
    }
    if extras:
        payload.update(extras)

    fpath.parent.mkdir(parents=True, exist_ok=True)
    fpath.write_text(
        _json.dumps(payload, ensure_ascii=False, indent=2,
                    default=lambda o: o.isoformat() if hasattr(o, "isoformat") else str(o)),
        encoding="utf-8",
    )
