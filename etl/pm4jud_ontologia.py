#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
PM4JUD-ONTOLOGIA  v2.0
================================================================================

Dissertação de Mestrado — PPGIa/PUCPR
Título: PM4JUD — Otimização Multiobjetivo com Mineração de Processos e
        Simulação no Contexto do Fluxo Processual em Gabinetes de Magistrado
Autor:  Luiz Claudio Soares de Almeida
Orient: Prof. Dr. Edson Emilio Scalabrin
Ano:    2026

Descrição
---------
Classe central que carrega os 7 módulos OWL/RDF da Ontologia PM4JUD e
disponibiliza métodos semânticos para o pipeline P1→P9.

A ontologia é TRANSVERSAL ao pipeline — cada etapa a utiliza para uma
finalidade distinta:

  P1  ETL        → Módulo 5: mapeia códigos TPU → nomes canônicos (MAPA_TPU)
  P2  REFINE_1   → Módulo 5: ancora Perfil 1 (D'Castro) em nomes canônicos;
                   Módulo 3: protege pares semanticamente distintos (Classes)
  P3  COMPLEMENT → Módulo 3: detecta classes prioritárias HC/RHC;
                   Módulo 5: identifica rótulos colegiados (Movimentos);
                   Módulo 7: gera eventos sintéticos C1–C16
  P4  REFINE_2   → Módulo 5: canonicalização semântica TPU + SAGWeb
  P5  PM         → Módulo 5: normaliza nomes antes do IMf;
                   Módulo 3: classifica traços prioritários (Classes)
  P6  LTLf       → Módulo 7: constraints Declare C1–C16 + Metas CNJ 1/2/4
  P7  DES        → Módulo 7: restrições hard/soft do RISTJ para SimPy
  P8  OPT        → Módulo 7: Metas CNJ como objetivos da otimização
  P9  STAT       → Módulo 7: definições formais das métricas (T̄, G, κ, η)

Técnica de punning (OWL 2 DL)
------------------------------
Cada código TPU é declarado simultaneamente como owl:Class e
owl:NamedIndividual — preserva a hierarquia taxonômica e habilita consultas
SPARQL como instâncias.
Referência: Lenzerini et al. (2021), AI v.292, DOI 10.1016/j.artint.2020.103432

Módulos OWL/RDF (diretório ontologia/)
---------------------------------------
  1  MNI_Core.owl                 — MNI 2.2.2 (entidades base)
  2  PM4JUD_Classes.owl           — Tabela de Classes TPU/CNJ (133 STJ)
  3  PM4JUD_Assuntos.owl          — Tabela de Assuntos TPU/CNJ (3.278 STJ)
  4  PM4JUD_Movimentos.owl        — Tabela de Movimentos TPU/CNJ (616 STJ)
  5  PM4JUD_Documentos.owl        — Tabela de Documentos TPU/CNJ (1.361 STJ)
  6  MNI_STJ.owl                  — Especialidades (Criminal/Cível/Trib./Prev.)
  7  PM4JUD.owl                   — Restrições Regimentais RISTJ + Metas CNJ

Uso
---
  from pm4jud_ontologia import OntologiaPM4JUD

  ont = OntologiaPM4JUD(Path("../ontologia"))
  ont.carregar()                        # carrega todos os 7 módulos

  mapa    = ont.mapa_tpu()              # {int: str} — P1, P2, P4, P5
  prio    = ont.classes_prioritarias()  # {str}      — P2, P3, P5
  coleg   = ont.rotulos_colegiada()     # {str}      — P3, P4
  c_rules = ont.constraints_ltlf()     # [dict]     — P3, P6
  metas   = ont.metas_cnj()            # [dict]     — P6, P8, P9
  specs   = ont.especialidades()        # {str: [int]} — P5, P6

Repositório: https://github.com/luizcsalmeida/pm4jud/tree/main/etl
================================================================================
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set

log = logging.getLogger("PM4JUD-Ontologia")

# ---------------------------------------------------------------------------
# Dependência opcional — rdflib
# ---------------------------------------------------------------------------
try:
    from rdflib import Graph, Namespace, URIRef
    from rdflib.namespace import OWL, RDF, RDFS, XSD
    RDFLIB_OK = True
except ImportError:
    RDFLIB_OK = False
    log.warning(
        "rdflib não disponível — Ontologia PM4JUD operará em modo fallback. "
        "Execute: pip install rdflib"
    )

# ---------------------------------------------------------------------------
# Namespaces da Ontologia PM4JUD
# ---------------------------------------------------------------------------
NS_STJ      = "http://www.stj.jus.br/mni/intercomunicacao#"
NS_CLASSES  = "http://www.pucpr.br/pm4jud/classes#"
NS_MOVS     = "http://www.pucpr.br/pm4jud/movimentos#"
NS_ASSUNTOS = "http://www.pucpr.br/pm4jud/assuntos#"
NS_DOCS     = "http://www.pucpr.br/pm4jud/documentos#"
NS_PM4JUD   = "http://www.pucpr.br/pm4jud#"
NS_MNI      = "http://www.cnj.jus.br/mni#"


# ---------------------------------------------------------------------------
# Classe principal
# ---------------------------------------------------------------------------

class OntologiaPM4JUD:
    """
    Camada semântica transversal ao pipeline PM4JUD.

    Carrega os 7 módulos OWL/RDF da Ontologia PM4JUD e expõe métodos de
    consulta SPARQL para cada etapa do pipeline P1→P9.  Usa lazy-loading
    por módulo e cache de resultados para evitar consultas repetidas.
    """

    # Mapeamento módulo → arquivo OWL relativo ao diretório de ontologias
    # Arquivos reais confirmados (mai/2026):
    ARQUIVOS: Dict[int, str] = {
        1: "MNI_Core.owl",            # Estrutura base MNI 2.2.2
        2: "MNI_STJ.owl",             # Especialização STJ do MNI
        3: "PM4JUD_Classes.owl",      # Tabela de Classes TPU/CNJ (133 STJ)
        4: "PM4JUD_Assuntos.owl",     # Tabela de Assuntos TPU/CNJ (3.278 STJ)
        5: "PM4JUD_Movimentos.owl",   # Tabela de Movimentos TPU/CNJ (619 STJ)
        6: "PM4JUD_Documentos.owl",   # Tabela de Documentos TPU/CNJ (1.361 STJ)
        7: "PM4JUD.owl",              # Restrições Regimentais RISTJ + Metas CNJ
    }

    # ---------------------------------------------------------------------------
    # Construtor
    # ---------------------------------------------------------------------------

    def __init__(self, ontologia_dir: Path) -> None:
        """
        Parameters
        ----------
        ontologia_dir : Path
            Diretório contendo os arquivos .owl dos 7 módulos.
            Padrão do projeto: <workspace>/ontologia/
        """
        self._dir    = Path(ontologia_dir)
        self._grafos: Dict[int, "Graph"] = {}   # cache de grafos rdflib
        self._cache:  Dict[str, object]  = {}   # cache de resultados SPARQL

        if not self._dir.exists():
            log.warning(
                "Diretório de ontologia não encontrado: %s — "
                "operando em modo fallback.", self._dir
            )

    # ---------------------------------------------------------------------------
    # Carregamento de módulos
    # ---------------------------------------------------------------------------

    def carregar(self, modulos: Optional[List[int]] = None) -> "OntologiaPM4JUD":
        """
        Carrega os módulos OWL especificados (padrão: todos os 7).

        Parameters
        ----------
        modulos : list[int], optional
            Números dos módulos a carregar (1–7).  Se None, carrega todos.

        Returns
        -------
        self (para encadeamento)
        """
        alvos = modulos or list(self.ARQUIVOS.keys())
        for n in alvos:
            self._carregar_modulo(n)
        return self

    def _carregar_modulo(self, n: int) -> Optional["Graph"]:
        """Carrega (e cacheia) o módulo n. Retorna o grafo ou None."""
        if n in self._grafos:
            return self._grafos[n]

        if not RDFLIB_OK:
            return None

        arquivo = self.ARQUIVOS.get(n)
        if not arquivo:
            log.warning("Módulo %d não reconhecido.", n)
            return None

        owl_path = self._dir / arquivo
        if not owl_path.exists():
            log.warning("Módulo %d não encontrado: %s", n, owl_path)
            return None

        try:
            g = Graph()
            g.parse(str(owl_path), format="xml")
            self._grafos[n] = g
            log.info(
                "Ontologia Módulo %d carregada: %s (%d triplas)",
                n, arquivo, len(g)
            )
            return g
        except Exception as exc:
            log.error("Erro ao carregar Módulo %d (%s): %s", n, arquivo, exc)
            return None

    def modulo(self, n: int) -> Optional["Graph"]:
        """Retorna o grafo do módulo n (carregando se necessário)."""
        return self._carregar_modulo(n)

    def status(self) -> Dict[str, object]:
        """Retorna o status de carga dos 7 módulos."""
        return {
            f"modulo_{n}": (
                f"{self.ARQUIVOS[n]} ({'ok' if n in self._grafos else 'nao carregado'})"
                + (f" — {len(self._grafos[n])} triplas" if n in self._grafos else "")
            )
            for n in self.ARQUIVOS
        }

    # ---------------------------------------------------------------------------
    # Módulo 4 — Movimentos Processuais TPU
    # Usado por: P1 ETL, P2 REFINE_1, P4 REFINE_2, P5 PM
    # ---------------------------------------------------------------------------

    def mapa_tpu(self) -> Dict[int, str]:
        """
        Módulo 4 → {codigo_tpu: nome_canonico}

        Mapeia códigos numéricos dos movimentos TPU/CNJ para seus nomes
        canônicos extraídos da ontologia (stj:tipoMovimento ou rdfs:label@pt).

        Preferência por stj:tipoMovimento (mais específico — distingue
        subtipo de HC concedido vs denegado vs não conhecido).

        Usado por: P1 ETL, P2 REFINE_1, P4 REFINE_2, P5 PM
        """
        cache_key = "mapa_tpu"
        if cache_key in self._cache:
            return self._cache[cache_key]

        g = self._carregar_modulo(5)
        if g is None:
            log.warning("Módulo 5 (Movimentos) indisponível — mapa_tpu() retorna fallback.")
            return self._mapa_tpu_fallback()

        sparql = """
            PREFIX stj:  <http://www.stj.jus.br/mni/intercomunicacao#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>

            SELECT ?codigo ?tipo ?label WHERE {
                ?mov stj:codigoMovimentoTPU ?codigo .
                OPTIONAL { ?mov stj:tipoMovimento ?tipo }
                OPTIONAL {
                    ?mov rdfs:label ?label .
                    FILTER(lang(?label) = "pt")
                }
            }
            ORDER BY xsd:integer(?codigo)
        """
        mapa: Dict[int, str] = {}
        try:
            for row in g.query(sparql):
                cod = int(row.codigo)
                if row.tipo and str(row.tipo).strip():
                    nome = _limpar_tipo_movimento(str(row.tipo))
                elif row.label and str(row.label).strip():
                    nome = str(row.label).strip()
                else:
                    continue
                if cod and nome:
                    mapa[cod] = nome
            log.info("mapa_tpu(): %d movimentos carregados do Módulo 5 (Movimentos).", len(mapa))
        except Exception as exc:
            log.error("mapa_tpu() SPARQL error: %s", exc)
            return self._mapa_tpu_fallback()

        self._cache[cache_key] = mapa
        return mapa

    @staticmethod
    def _mapa_tpu_fallback() -> Dict[int, str]:
        """Fallback mínimo quando Módulo 4 não está disponível."""
        return {
            3:     "Decisão",
            14:    "Despacho",
            26:    "Distribuído",
            92:    "Publicado no DJe",
            132:   "Recebimento",
            193:   "Julgado",
            339:   "Liminar deferida",
            340:   "Liminar indeferida",
            443:   "HC concedido",
            447:   "HC denegado",
            451:   "HC concedido parcialmente",
            1061:  "Disponibilizado no DJe",
            11383: "Ato ordinatório",
            12458: "HC não conhecido",
            12475: "HC concedido de ofício",
            14093: "Voto do relator proferido",
        }

    def pares_semanticamente_distintos(self) -> FrozenSet[FrozenSet[str]]:
        """
        Módulo 4 → pares de atividades que NÃO devem ser agrupadas no
        Perfil 1 do D'Castro, apesar de terem texto similar.

        Exemplo: {"HC concedido", "HC denegado"} — textos similares mas
        resultados opostos; agrupar destruiria a análise de conformidade LTLf.

        Usado por: P2 REFINE_1
        """
        cache_key = "pares_distintos"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Extrair da ontologia: movimentos com mesmo rdfs:label mas
        # diferentes stj:tipoMovimento (indica subtipos distintos).
        g = self._carregar_modulo(5)
        pares: Set[FrozenSet[str]] = set()

        if g is not None:
            sparql = """
                PREFIX stj:  <http://www.stj.jus.br/mni/intercomunicacao#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

                SELECT ?tipo1 ?tipo2 WHERE {
                    ?m1 stj:tipoMovimento ?tipo1 .
                    ?m2 stj:tipoMovimento ?tipo2 .
                    ?m1 rdfs:label ?label .
                    ?m2 rdfs:label ?label .
                    FILTER(?m1 != ?m2 && str(?tipo1) < str(?tipo2))
                }
            """
            try:
                for row in g.query(sparql):
                    t1 = _limpar_tipo_movimento(str(row.tipo1))
                    t2 = _limpar_tipo_movimento(str(row.tipo2))
                    if t1 and t2 and t1 != t2:
                        pares.add(frozenset({t1, t2}))
            except Exception as exc:
                log.warning("pares_semanticamente_distintos() SPARQL error: %s", exc)

        # Protegidos explícitos — resultados de HC nunca devem ser agrupados
        protegidos_fixos = [
            {"HC concedido", "HC denegado"},
            {"HC concedido", "HC não conhecido"},
            {"HC concedido", "HC concedido parcialmente"},
            {"HC denegado", "HC não conhecido"},
            {"HC concedido de ofício", "HC denegado"},
            {"Liminar deferida", "Liminar indeferida"},
            {"Publicado no DJe", "Disponibilizado no DJe"},
        ]
        for par in protegidos_fixos:
            pares.add(frozenset(par))

        resultado = frozenset(pares)
        self._cache[cache_key] = resultado
        log.info(
            "pares_semanticamente_distintos(): %d pares protegidos.", len(resultado)
        )
        return resultado

    def rotulos_colegiada(self) -> Set[str]:
        """
        Módulo 4 → nomes de atividades que caracterizam decisões colegiadas
        (acórdão, voto do relator, sessão de julgamento).

        Usado por: P3 COMPLEMENT, P4 REFINE_2
        """
        cache_key = "rotulos_colegiada"
        if cache_key in self._cache:
            return self._cache[cache_key]

        g = self._carregar_modulo(5)
        if g is None:
            return self._rotulos_colegiada_fallback()

        sparql = """
            PREFIX stj:  <http://www.stj.jus.br/mni/intercomunicacao#>
            PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>

            SELECT ?tipo WHERE {
                ?mov stj:codigoMovimentoTPU ?codigo ;
                     stj:tipoMovimento ?tipo .
                VALUES ?codigo {
                    "14093"^^xsd:integer  "193"^^xsd:integer
                    "443"^^xsd:integer    "447"^^xsd:integer
                    "451"^^xsd:integer    "12458"^^xsd:integer
                    "12475"^^xsd:integer  "417"^^xsd:integer
                }
            }
        """
        rotulos: Set[str] = set()
        try:
            for row in g.query(sparql):
                nome = _limpar_tipo_movimento(str(row.tipo))
                if nome:
                    rotulos.add(nome)
        except Exception as exc:
            log.warning("rotulos_colegiada() SPARQL error: %s", exc)

        if not rotulos:
            rotulos = self._rotulos_colegiada_fallback()

        self._cache[cache_key] = rotulos
        log.info("rotulos_colegiada(): %d rótulos carregados.", len(rotulos))
        return rotulos

    @staticmethod
    def _rotulos_colegiada_fallback() -> Set[str]:
        return {
            "Voto do relator proferido",
            "Julgado",
            "HC concedido",
            "HC denegado",
            "HC concedido parcialmente",
            "HC não conhecido",
            "HC concedido de ofício",
            "Inclusão em pauta",
        }

    # ---------------------------------------------------------------------------
    # Módulo 2 — Classes Processuais TPU
    # Usado por: P2 REFINE_1, P3 COMPLEMENT, P5 PM
    # ---------------------------------------------------------------------------

    def classes_prioritarias(self) -> Set[str]:
        """
        Módulo 2 → códigos string das classes processuais com prioridade
        regimental (FamiliaHabeasCorpus: HC, RHC e derivados).

        Arts. 91 e 202 RISTJ: independem de pauta.

        Usado por: P2 REFINE_1, P3 COMPLEMENT, P5 PM
        """
        cache_key = "classes_prioritarias"
        if cache_key in self._cache:
            return self._cache[cache_key]

        g = self._carregar_modulo(3)
        if g is None:
            return self._classes_prioritarias_fallback()

        sparql = f"""
            PREFIX pm4jud: <{NS_CLASSES}>
            PREFIX rdf:    <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

            SELECT DISTINCT ?codigo WHERE {{
                ?c rdf:type pm4jud:FamiliaHabeasCorpus ;
                   pm4jud:codigoTPU ?codigo .
            }}
        """
        codigos: Set[str] = set()
        try:
            for row in g.query(sparql):
                codigos.add(str(row.codigo))
        except Exception as exc:
            log.warning("classes_prioritarias() SPARQL error: %s", exc)

        if not codigos:
            codigos = self._classes_prioritarias_fallback()

        self._cache[cache_key] = codigos
        log.info(
            "classes_prioritarias(): %d classes carregadas do Módulo 3 (Classes).", len(codigos)
        )
        return codigos

    @staticmethod
    def _classes_prioritarias_fallback() -> Set[str]:
        """5 códigos confirmados no SGT/STJ (versão 09/04/2026)."""
        return {"1720", "1013", "1722", "1064", "15224"}

    # ---------------------------------------------------------------------------
    # Módulo 3 — Assuntos Processuais
    # Usado por: P5 PM, P6 LTLf
    # ---------------------------------------------------------------------------

    def especialidades(self) -> Dict[str, List[int]]:
        """
        Módulo 3 + 6 → {especialidade: [codigos_assunto_tpu]}

        Mapeia cada especialidade (Criminal, Cível, Tributário, Previdenciário)
        ao conjunto de códigos de assunto TPU que a compõem.

        Usado por: P5 PM, P6 LTLf
        """
        cache_key = "especialidades"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Carrega módulo 3 (assuntos) e 6 (especialidades)
        g3 = self._carregar_modulo(4)
        g6 = self._carregar_modulo(3)

        if g3 is None or g6 is None:
            result = self._especialidades_fallback()
            self._cache[cache_key] = result
            return result

        sparql = f"""
            PREFIX pm4jud3: <{NS_ASSUNTOS}>
            PREFIX pm4jud6: <http://www.pucpr.br/pm4jud/especialidades#>
            PREFIX rdf:     <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

            SELECT ?especialidade ?codigo WHERE {{
                ?assunto pm4jud3:codigoTPU ?codigo ;
                         pm4jud3:pertenceAEspecialidade ?esp .
                ?esp rdfs:label ?especialidade .
            }}
        """
        result: Dict[str, List[int]] = {}
        try:
            # Combina os dois grafos para a query
            g_combined = g3 + g6
            for row in g_combined.query(sparql):
                esp = str(row.especialidade).strip()
                cod = int(row.codigo)
                result.setdefault(esp, []).append(cod)
            log.info(
                "especialidades(): %d especialidades, %d assuntos totais.",
                len(result),
                sum(len(v) for v in result.values()),
            )
        except Exception as exc:
            log.warning("especialidades() SPARQL error: %s", exc)
            result = self._especialidades_fallback()

        self._cache[cache_key] = result
        return result

    @staticmethod
    def _especialidades_fallback() -> Dict[str, List[int]]:
        """Fallback mínimo para especialidades."""
        return {
            "Criminal":       [0],
            "Cível":          [0],
            "Tributário":     [0],
            "Previdenciário": [0],
        }

    # ---------------------------------------------------------------------------
    # Módulo 7 — Restrições Regimentais e Metas CNJ
    # Usado por: P3 COMPLEMENT, P6 LTLf, P7 DES, P8 OPT, P9 STAT
    # ---------------------------------------------------------------------------

    def constraints_ltlf(self) -> List[Dict]:
        """
        Módulo 7 → [{'id': 'C1', 'artigo': 'Art.91,I', 'prazo_dias': 2,
                      'nivel': 1, 'pattern': 'response', 'atividade': '...'}]

        Regras regimentais formalizadas como constraints Declare/LTLf.
        C1–C16: cada regra tem padrão temporal, prazo e nível de verificação.

        Nível 1: verificável com dados públicos DATAJUD (Fase 1).
        Nível 2: verificável com dados SAGWeb (Fase 2).

        Usado por: P3 COMPLEMENT (gera eventos sintéticos),
                   P6 LTLf (verifica conformidade),
                   P7 DES (restrições hard/soft SimPy)
        """
        cache_key = "constraints_ltlf"
        if cache_key in self._cache:
            return self._cache[cache_key]

        g = self._carregar_modulo(7)
        if g is None:
            result = self._constraints_ltlf_fallback()
            self._cache[cache_key] = result
            return result

        # SPARQL PM4JUD.owl v2.0:
        # - Âncora: pm4jud:codigoRegra — único em todos C1-C16
        # - regraAtiva removido (inconsistente xsd:boolean entre versões rdflib)

        # SPARQL PM4JUD.owl v2.0 — âncora por padrão de URI.
        # O rdflib não parseia corretamente data property assertions no namespace
        # default do RDF/XML (pm4jud:codigoRegra retorna 0 triplas como predicado).
        # Solução: localizar os indivíduos RegraC* pelo prefixo de URI
        # (STRSTARTS) — independe de data properties sendo corretamente parseadas.
        # REPLACE extrai o ID canônico: "http://...#RegraC1" → "C1".
        # DISTINCT necessário porque cada indivíduo pode ter múltiplos rdf:type.
        sparql = f"""
            PREFIX rdf:    <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs:   <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX pm4jud: <{NS_PM4JUD}>

            SELECT DISTINCT ?regra ?artigo ?prazo ?nivel ?padrao ?label
                            ?ativ_ant ?ativ_cons ?sag_ant ?sag_cons WHERE {{
                ?regra rdf:type ?tipo .
                FILTER(STRSTARTS(STR(?regra), "{NS_PM4JUD}RegraC"))
                OPTIONAL {{ ?regra pm4jud:temFundamentoNoArtigo     ?artigo   }}
                OPTIONAL {{ ?regra pm4jud:prazoMaxDiasUteis         ?prazo    }}
                OPTIONAL {{ ?regra pm4jud:nivelVerificacao           ?nivel    }}
                OPTIONAL {{ ?regra pm4jud:padraoLTLf                 ?padrao   }}
                OPTIONAL {{ ?regra rdfs:label                        ?label    }}
                OPTIONAL {{ ?regra pm4jud:temAtividadeAntecedente    ?ativ_ant }}
                OPTIONAL {{ ?regra pm4jud:temAtividadeConsequente    ?ativ_cons}}
                OPTIONAL {{ ?regra pm4jud:temAtividadeSAGWebAntecedente ?sag_ant }}
                OPTIONAL {{ ?regra pm4jud:temAtividadeSAGWebConsequente ?sag_cons}}
            }}
            ORDER BY ?regra
        """
        constraints: List[Dict] = []
        try:
            seen_ids: set = set()
            for row in g.query(sparql):
                # Extrai ID canônico da URI: "http://...#RegraC10" → "C10"
                uri_str = str(row.regra)
                codigo  = uri_str.split("#Regra")[-1] if "#Regra" in uri_str else ""
                if not codigo or codigo in seen_ids:
                    continue      # pula duplicatas do DISTINCT por múltiplos rdf:type
                seen_ids.add(codigo)
                prazo_raw = int(row.prazo) if row.prazo is not None else None
                prazo_dias = prazo_raw if (prazo_raw and prazo_raw > 0) else None
                # Resolver URI do movimento → nome canônico via sufixo numérico
                def _uri_to_str(val):
                    return str(val) if val else None

                constraints.append({
                    "id":         codigo,
                    "uri":        uri_str,
                    "artigo":     str(row.artigo)  if row.artigo  else "",
                    "prazo_dias": prazo_dias,
                    "nivel":      int(row.nivel)   if row.nivel   else 1,
                    "padrao":     str(row.padrao)  if row.padrao  else "existence",
                    "label":      str(row.label)   if row.label   else codigo,
                    # Atividades: URIs de MovimentoProcessual ou strings SAGWeb
                    "atividade_a_uri": _uri_to_str(row.ativ_ant),
                    "atividade_b_uri": _uri_to_str(row.ativ_cons),
                    "atividade_a_sag": _uri_to_str(row.sag_ant),
                    "atividade_b_sag": _uri_to_str(row.sag_cons),
                })
            log.info("constraints_ltlf(): %d regras carregadas do Módulo 7.", len(constraints))
        except Exception as exc:
            log.warning("constraints_ltlf() SPARQL error: %s", exc)
            constraints = self._constraints_ltlf_fallback()

        if not constraints:
            constraints = self._constraints_ltlf_fallback()

        self._cache[cache_key] = constraints
        return constraints

    @staticmethod
    def _constraints_ltlf_fallback() -> List[Dict]:
        """Fallback com as regras principais do RISTJ para HC."""
        return [
            {"id": "C1",  "artigo": "Art.91,I",   "prazo_dias": None, "nivel": 1,
             "padrao": "existence", "label": "HC independe de pauta"},
            {"id": "C2",  "artigo": "Art.202",     "prazo_dias": 2,   "nivel": 2,
             "padrao": "response",  "label": "MP manifesta em 2 dias"},
            {"id": "C3",  "artigo": "Art.110,I",   "prazo_dias": 10,  "nivel": 2,
             "padrao": "bounded",   "label": "Decisão interlocutória em 10 dias"},
            {"id": "C4",  "artigo": "Art.110,III", "prazo_dias": 30,  "nivel": 2,
             "padrao": "bounded",   "label": "Visto do relator em 30 dias"},
            {"id": "C5",  "artigo": "Art.111",     "prazo_dias": 5,   "nivel": 2,
             "padrao": "bounded",   "label": "Servidor executa ato em 5 dias"},
        ]

    def metas_cnj(self) -> List[Dict]:
        """
        Módulo 7 → [{'id': 'Meta1', 'descricao': '...', 'formula': '...'}]

        Metas Nacionais CNJ 1, 2 e 4 formalizadas como objetivos do MOOP.

        Usado por: P6 LTLf (calcula η),
                   P8 OPT (função objetivo f4),
                   P9 STAT (definição formal de η)
        """
        cache_key = "metas_cnj"
        if cache_key in self._cache:
            return self._cache[cache_key]

        g = self._carregar_modulo(7)
        if g is None:
            result = self._metas_cnj_fallback()
            self._cache[cache_key] = result
            return result

        sparql = f"""
            PREFIX pm4jud: <{NS_PM4JUD}>
            PREFIX rdf:    <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs:   <http://www.w3.org/2000/01/rdf-schema#>

            SELECT ?meta ?label ?indicador WHERE {{
                ?meta rdf:type pm4jud:MetaNacionalCNJ .
                OPTIONAL {{ ?meta rdfs:label        ?label     }}
                OPTIONAL {{ ?meta pm4jud:temIndicador ?indicador }}
            }}
            ORDER BY ?label
        """
        metas: List[Dict] = []
        try:
            for row in g.query(sparql):
                uri_str = str(row.meta)
                # C7/C8/C9 são MetaNacionalCNJ no OWL mas são constraints RISTJ,
                # não Metas CNJ propriamente ditas. Filtrar por URI (Meta*_CNJ).
                if "RegraC" in uri_str:
                    continue
                metas.append({
                    "uri":       uri_str,
                    "label":     str(row.label)     if row.label     else "",
                    "indicador": str(row.indicador) if row.indicador else "",
                })
            log.info("metas_cnj(): %d metas carregadas do Módulo 7.", len(metas))
        except Exception as exc:
            log.warning("metas_cnj() SPARQL error: %s", exc)
            metas = self._metas_cnj_fallback()

        if not metas:
            metas = self._metas_cnj_fallback()

        self._cache[cache_key] = metas
        return metas

    @staticmethod
    def _metas_cnj_fallback() -> List[Dict]:
        return [
            {"uri": f"{NS_PM4JUD}Meta1_CNJ", "label": "Meta1_CNJ",
             "indicador": "J_sim / D_sim >= 1.00"},
            {"uri": f"{NS_PM4JUD}Meta2_CNJ", "label": "Meta2_CNJ",
             "indicador": "J_acervo / N_acervo >= 1.00"},
            {"uri": f"{NS_PM4JUD}Meta4_CNJ", "label": "Meta4_CNJ",
             "indicador": "J_prior / N_prior >= 0.90"},
        ]

    # ---------------------------------------------------------------------------
    # Traversal hierárquico — rdfs:subClassOf*
    # ---------------------------------------------------------------------------

    def descendentes_hierarquia(self,
                                 uri_raiz: str,
                                 modulo: int = 2) -> "Set[str]":
        """
        Retorna todos os URIs subclasses de uri_raiz via rdfs:subClassOf*.
        Traversal completo em qualquer profundidade (SPARQL 1.1 property path).
        Parâmetros: modulo = 1 (Classes) | 2 (Assuntos) | 3 (Movimentos).
        Retorna set vazio se o módulo não estiver carregado.
        """
        g = self._carregar_modulo(modulo)
        if g is None:
            return set()
        sparql = """
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT DISTINCT ?descendente WHERE {
                ?descendente rdfs:subClassOf* <%s> .
            }
        """ % uri_raiz
        try:
            return {str(row.descendente) for row in g.query(sparql)}
        except Exception as exc:
            log.warning("descendentes_hierarquia() SPARQL error: %s", exc)
            return set()

    def codigos_tpu_da_familia(self,
                                uri_raiz: str,
                                modulo: int = 2) -> "Set[int]":
        """
        Retorna o conjunto de códigos TPU (int) de todos os assuntos/classes
        descendentes de uri_raiz via rdfs:subClassOf*, incluindo a própria raiz.

        Implementação em DUAS queries SPARQL sequenciais — sem fallback Python.
        Necessário porque o rdflib 7.x não combina property paths (rdfs:subClassOf*)
        com BIND na mesma query (retorna 0 resultados silenciosamente).

        Query 1 — rdfs:subClassOf* → URIs dos descendentes (property path isolado)
        Query 2 — VALUES + BIND + REPLACE(STR(?uri), "^.+_", "") → códigos numéricos

        Padrão de URI esperado: http://...#Assunto_287  →  "287" → 287
        URIs sem underscore (ex: FamiliaHabeasCorpus) produzem ?raw = URI completa,
        e FILTER(isLiteral(?codigo)) as descarta silenciosamente.

        Retorna set vazio se o módulo não estiver carregado ou se a raiz
        não existir no grafo. Não há fallback Python — se o módulo OWL não
        estiver disponível, PM4JUD-LTLf opera sem classificação semântica.
        """
        g = self._carregar_modulo(modulo)
        if g is None:
            return set()

        # Query 1: traversal hierárquico via property path
        # Isolado do BIND para contornar limitação do rdflib com property paths
        sparql_hierarquia = """
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT DISTINCT ?descendente WHERE {
                ?descendente rdfs:subClassOf* <%s> .
            }
        """ % uri_raiz
        try:
            descendentes = [str(row.descendente)
                            for row in g.query(sparql_hierarquia)]
        except Exception as exc:
            log.warning("codigos_tpu_da_familia() Query 1 error: %s", exc)
            return set()

        if not descendentes:
            return set()

        # Query 2: extrair código numérico da URI via SPARQL VALUES + BIND+REPLACE
        # REPLACE(STR(?uri), "^.+_", "") → sufixo após o último underscore
        # xsd:integer() → conversão; FILTER(isLiteral) → descarta não-numéricos
        values_clause = " ".join(f"<{u}>" for u in descendentes)
        sparql_codigos = """
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
            SELECT DISTINCT ?codigo WHERE {
                VALUES ?uri { %s }
                BIND(REPLACE(STR(?uri), "^.+_", "") AS ?raw)
                BIND(xsd:integer(?raw) AS ?codigo)
                FILTER(isLiteral(?codigo))
            }
        """ % values_clause
        try:
            codigos: Set[int] = set()
            for row in g.query(sparql_codigos):
                try:
                    codigos.add(int(row.codigo))
                except (ValueError, TypeError):
                    pass
            return codigos
        except Exception as exc:
            log.warning("codigos_tpu_da_familia() Query 2 error: %s", exc)
            return set()

    def categorias_assunto(self) -> "Dict[str, Set[int]]":
        """
        Retorna {categoria: Set[cod_tpu]} usando rdfs:subClassOf* sobre o
        módulo de Assuntos.  Baseado na taxonomia TPU/CNJ 09/04/2026.

        Categorias:
          "crimes_adm_publica"  → Meta CNJ 4a (C9 do PM4JUD)
          "crimes_patrimonio"   → contexto criminal
          "direito_penal"       → especialidade Criminal
          "direito_civil"       → especialidade Cível
          "tributario"          → especialidade Tributário
          "previdenciario"      → especialidade Previdenciário

        Para obter as URIs raiz corretas, execute no Protégé sobre
        PM4JUD_Assuntos.owl:
            SELECT ?c ?l WHERE { ?c rdfs:label ?l .
                FILTER(CONTAINS(LCASE(str(?l)), "administra")) }

        Retorna dict com sets vazios se o módulo Assuntos não estiver
        disponível (não falha — permite execução em Fase 1 sem o módulo).
        """
        cache_key = "categorias_assunto"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # URIs raiz confirmadas via buscar_uris_assuntos.py (mai/2026)
        # no PM4JUD_Assuntos.owl (34.619 triplas, SGT/STJ 09/04/2026)
        #
        # Raízes principais da árvore TPU:
        #   Assunto_287  = DIREITO PENAL (raiz penal geral)
        #   Assunto_14   = DIREITO TRIBUTÁRIO
        #   Assunto_195  = DIREITO PREVIDENCIÁRIO
        #   Assunto_899  = DIREITO CIVIL
        #
        # crimes_adm_publica (Meta CNJ 4a) abrange DOIS subconjuntos:
        #   Assunto_3923 = Crimes contra a Administração Pública (filho de Assunto_287)
        #   Assunto_10011 = Improbidade Administrativa (Direito Administrativo)
        #   Assunto_4263  = Ação Penal (originária — competência STJ)
        # Estratégia: computar os três e fazer a união dos sets.
        #
        # crimes_patrimonio: Assunto_3415 (filho de Assunto_287)
        RAIZES: Dict[str, str] = {
            "direito_penal":      f"{NS_ASSUNTOS}Assunto_287",
            "crimes_patrimonio":  f"{NS_ASSUNTOS}Assunto_3415",
            "direito_civil":      f"{NS_ASSUNTOS}Assunto_899",
            "tributario":         f"{NS_ASSUNTOS}Assunto_14",
            "previdenciario":     f"{NS_ASSUNTOS}Assunto_195",
        }
        # Meta CNJ 4a: crimes_adm_publica = união de três subárvores
        RAIZES_ADM_PUBLICA: List[str] = [
            f"{NS_ASSUNTOS}Assunto_3923",   # Crimes contra a Adm. Pública (penal)
            f"{NS_ASSUNTOS}Assunto_10011",  # Improbidade Administrativa
            f"{NS_ASSUNTOS}Assunto_4263",   # Ação Penal Originária (competência STJ)
        ]
        resultado: Dict[str, Set[int]] = {}
        for cat, uri in RAIZES.items():
            codigos = self.codigos_tpu_da_familia(uri, modulo=4)
            resultado[cat] = codigos
            if codigos:
                log.info("categorias_assunto: %-22s → %d códigos TPU", cat, len(codigos))
            else:
                log.debug("categorias_assunto: %-22s → módulo Assuntos não carregado "
                          "(fallback por especialidades())", cat)

        # crimes_adm_publica (Meta CNJ 4a): união de três subárvores
        set_adm: Set[int] = set()
        for uri in RAIZES_ADM_PUBLICA:
            set_adm |= self.codigos_tpu_da_familia(uri, modulo=4)
        resultado["crimes_adm_publica"] = set_adm
        if set_adm:
            log.info("categorias_assunto: %-22s → %d códigos TPU (3 subárvores)",
                     "crimes_adm_publica", len(set_adm))
        else:
            log.debug("categorias_assunto: crimes_adm_publica → módulo não carregado")

        self._cache[cache_key] = resultado
        return resultado

    def restricoes_hard(self) -> List[Dict]:
        """
        Módulo 7 → constraints de nível 1 e 2 que são rígidas (hard).
        Violação invalida uma solução no DES/OPT.

        Usado por: P7 DES, P8 OPT
        """
        return [c for c in self.constraints_ltlf() if c.get("nivel", 1) <= 2]

    def restricoes_soft(self) -> List[Dict]:
        """
        Módulo 7 → constraints cujas violações penalizam a função objetivo
        mas não invalidam a solução.

        Usado por: P7 DES, P8 OPT
        """
        hard_ids = {c["id"] for c in self.restricoes_hard()}
        return [c for c in self.constraints_ltlf() if c["id"] not in hard_ids]

    # ---------------------------------------------------------------------------
    # Utilitário: resolução de nome canônico para uma atividade
    # ---------------------------------------------------------------------------

    def resolver_nome(self, conceito: str) -> str:
        """
        Dado um nome de atividade do log XES, tenta resolver para o nome
        canônico definido na ontologia.  Usado por P5 PM antes do IMf.

        Para atividades TPU: busca no mapa_tpu() por correspondência exata
        de nome ou por código embutido no atributo pm4jud:codigo_tpu.
        Para atividades SAGWeb (sem código TPU): retorna o nome sem alteração.

        Parameters
        ----------
        conceito : str
            Valor do atributo concept:name do evento XES.

        Returns
        -------
        str
            Nome canônico da ontologia, ou o conceito original se não mapeado.
        """
        mapa = self.mapa_tpu()
        # Se já é um nome canônico (está nos valores do mapa), retorna direto
        if conceito in mapa.values():
            return conceito
        # Tenta match por similaridade simples (lower-stripped)
        conceito_norm = conceito.strip().lower()
        for nome in mapa.values():
            if nome.strip().lower() == conceito_norm:
                return nome
        # Não mapeado — atividade SAGWeb ou rótulo desconhecido
        return conceito

    # ---------------------------------------------------------------------------
    # Representação
    # ---------------------------------------------------------------------------

    def __repr__(self) -> str:
        carregados = list(self._grafos.keys())
        return (
            f"OntologiaPM4JUD(dir={self._dir}, "
            f"modulos_carregados={carregados})"
        )


# ---------------------------------------------------------------------------
# Utilitários de limpeza de rótulos (compartilhados com pm4jud_vocab)
# ---------------------------------------------------------------------------

def _limpar_tipo_movimento(tipo: str) -> str:
    """
    Remove templates #{...} e #(...) do atributo stj:tipoMovimento da
    ontologia e normaliza o rótulo resultante para uso como concept:name
    no log XES.

    Exemplos:
      "Concedido o HC a #{nome_da_parte}" → "Concedido o HC"
      "Publicado #{ato} em #{data}."      → "Publicado"
    """
    limpo = re.sub(r"#\{[^}]+\}", "", tipo)
    limpo = re.sub(r"#\([^)]+\)", "", limpo)
    # Remove preposições/artigos isolados no final
    limpo = re.sub(
        r"\s+(a|ao|aos|de|do|da|dos|das|em|no|na|nos|nas|para|por|pelo|pela|o|os)\s*$",
        "", limpo.strip(), flags=re.IGNORECASE
    )
    limpo = re.sub(r"\s+", " ", limpo)
    limpo = re.sub(r"[\s.,;:]+$", "", limpo.strip())
    return limpo if limpo else tipo


# ---------------------------------------------------------------------------
# Fábrica conveniente
# ---------------------------------------------------------------------------

def carregar_ontologia(
    ontologia_dir: Path,
    modulos: Optional[List[int]] = None,
) -> OntologiaPM4JUD:
    """
    Fábrica conveniente: cria OntologiaPM4JUD, carrega os módulos e retorna.

    Uso padrão em cada programa do pipeline:

        from pm4jud_ontologia import carregar_ontologia
        ont = carregar_ontologia(args.ontologia)

    Parameters
    ----------
    ontologia_dir : Path
        Diretório com os arquivos .owl dos 7 módulos.
    modulos : list[int], optional
        Módulos a carregar.  None = todos os 7.

    Returns
    -------
    OntologiaPM4JUD instanciada e com os módulos solicitados carregados.
    """
    ont = OntologiaPM4JUD(ontologia_dir)
    ont.carregar(modulos)
    return ont
