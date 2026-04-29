# PM4JUD — Process Mining for Judicial Decision-Making

> **Framework integrado de Mineração de Processos, Simulação Computacional e Otimização Multiobjetivo para apoio à tomada de decisão operacional em gabinetes de magistrado do Superior Tribunal de Justiça**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PM4Py](https://img.shields.io/badge/PM4Py-2.7%2B-orange)](https://pm4py.fit.fraunhofer.de/)
[![SimPy](https://img.shields.io/badge/SimPy-4.x-green)](https://simpy.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Em%20desenvolvimento-yellow)]()
[![PPGIa/PUCPR](https://img.shields.io/badge/PPGIa-PUCPR-darkblue)](https://www.pucpr.br/escola-politecnica/mestrado-doutorado/informatica-aplicada/)

---

## Sobre o projeto

O **PM4JUD** é um framework científico desenvolvido como dissertação de mestrado no **Programa de Pós-Graduação em Informática Aplicada (PPGIa)** da **Pontifícia Universidade Católica do Paraná (PUCPR)**, sob orientação do **Prof. Dr. Edson Emilio Scalabrin**.

O framework adapta o **PM4SOS** (Ferronato, 2022) — originalmente aplicado a centros cirúrgicos hospitalares — ao domínio judicial, com foco nos gabinetes de ministros da **3.ª Seção do Superior Tribunal de Justiça (STJ)**, competente para matéria penal e processual penal. O PM4JUD integra três disciplinas para produzir recomendações de redistribuição processual baseadas em evidências:

- **Mineração de Processos** — descoberta automática do fluxo real de tramitação processual
- **Simulação Computacional** — geração e avaliação de cenários alternativos de alocação
- **Otimização Multiobjetivo** — comparação experimental entre NSGA-II, AMGA2 e SPEA2

---

## Problema de pesquisa

> *Como diferentes configurações de otimização multiobjetivo (NSGA-II, AMGA2 e SPEA2) se comportam na redução do tempo de julgamento em gabinetes de magistrados, por meio de experimentação controlada baseada em modelos de simulação gerados a partir de logs de eventos, considerando a especialização por assunto processual, as restrições regimentais do STJ e o cumprimento das Metas Nacionais do CNJ?*

---

## Pipeline de 5 fases

```
DATAJUD/CNJ ──► Fase 1 ──► Fase 2 ──► Fase 3 ──► Fase 4 ──► Fase 5 ──► Decisão
               Preparação  Mineração  Simulação  Otimização  Visual.   Magistrado
```

| Fase | Descrição | Programas |
|------|-----------|-----------|
| **1 — Preparação** | Extração do DATAJUD, tratamento de complexidade, complementação para LTLf | P1, P2, Sim2Log, Complement |
| **2 — Mineração** | Descoberta do modelo AS-IS, modelo organizacional, parâmetros, estado corrente, verificação LTLf | P3, P4b |
| **3 — Simulação** | Modelo DES M/M/c com prioridade HC, verificação, validação, gatilho | P5 |
| **4 — Otimização** | CBR + NSGA-II/AMGA2/SPEA2, 90 replicações, fronteira de Pareto | P6, P7 |
| **5 — Visualização** | Dashboard Nível 1 (DATAJUD) e Nível 2 (SAGWeb), configurações candidatas | VIZ |

> A **Ontologia PM4JUD** (7 módulos OWL/RDF, baseados em MNI/CNJ e TPU/CNJ) opera como camada semântica transversal às Fases 1 e 2.

---

## Artefatos computacionais

| Programa | Módulo | Descrição |
|----------|--------|-----------|
| `P1` | [`etl/pm4jud_etl.py`](etl/pm4jud_etl.py) | Extração DATAJUD → XES; filtro por ministro relator |
| `P2` | [`dcastro/pm4jud_dcastro.py`](dcastro/pm4jud_dcastro.py) | Tratamento de complexidade: 3 perfis D'Castro + NLP |
| `P3` | [`pm/pm4jud_pm.py`](pm/pm4jud_pm.py) | IMf (k=0.2), DFG, modelo organizacional, parâmetros |
| `P4a` | [`complement/pm4jud_complement.py`](complement/pm4jud_complement.py) | Injeção de atributos `[SIM-LTLf]` nos traços reais |
| `P4b` | [`ltlf/pm4jud_ltlf.py`](ltlf/pm4jud_ltlf.py) | Verificação Declare — Metas CNJ + RISTJ arts. 91/110/111 |
| `P5` | [`des/pm4jud_des.py`](des/pm4jud_des.py) | Modelo DES em SimPy; verificação e validação |
| `P6` | [`opt/pm4jud_opt.py`](opt/pm4jud_opt.py) | 90 execuções: GC + NSGA-II + AMGA2 + SPEA2 |
| `P7` | [`stat/pm4jud_stat.py`](stat/pm4jud_stat.py) | Shapiro-Wilk, ANOVA/Kruskal-Wallis, Bonferroni |
| — | [`sim2log/pm4jud_sim2log.py`](sim2log/pm4jud_sim2log.py) | Geração de eventos `[SIM-DES]` para simulação |
| — | [`viz/pm4jud_viz.py`](viz/pm4jud_viz.py) | Dashboard Nível 1 e 2 (PDF + HTML interativo) |

---

## Dados — estratégia bifásica

### Fase 1 (atual) — dados públicos
- **Fonte:** API pública DATAJUD/CNJ (Resolução CNJ nº 331/2020)
- **Escopo:** acervo completo dos gabinetes dos ministros Reynaldo Soares da Fonseca e Joel Ilan Paciornik (5.ª Turma) e Rogerio Schietti Cruz (6.ª Turma)
- **Período:** janeiro/2023 a dezembro/2024
- **Aprovação ética:** não requerida (dados públicos)
- **Acesso:** chave pública emitida pelo DPJ/CNJ em https://datajud-wiki.cnj.jus.br/api-publica/

### Fase 2 (futura) — dados operacionais
- **Fonte:** SAGWeb/STJ (Sistema de Automação de Gabinetes Web)
- **Conteúdo:** logs de ações sistêmicas dos assessores; movimentos internos; escaninhos
- **Condição:** autorização formal do STJ + aprovação CEP via Plataforma Brasil
- **Status:** em tramitação

> ⚠️ Os dados do SAGWeb **não são públicos** e não estão disponíveis neste repositório. Os eventos sintéticos gerados pela PM4JUD-Sim2Log são identificados pela marcação `[SIM-DES]`; os eventos complementados para verificação LTLf pela marcação `[SIM-LTLf]`. As duas marcações não devem ser confundidas nem usadas de forma cruzada.

---

## Gabinetes piloto

| Gabinete | Ministro | Turma | Seção |
|----------|----------|-------|-------|
| GAB-1 | Reynaldo Soares da Fonseca | 5.ª Turma | 3.ª Seção (Criminal) |
| GAB-2 | Joel Ilan Paciornik | 5.ª Turma | 3.ª Seção (Criminal) |
| GAB-3 | Rogerio Schietti Cruz | 6.ª Turma | 3.ª Seção (Criminal) |

---

## Indicadores de desempenho

| Símbolo | Indicador | Unidade |
|---------|-----------|---------|
| **T̄** | Tempo médio de julgamento | Dias |
| **G** | Coeficiente de Gini de balanceamento de carga | Adimensional [0,1] |
| **κ** | Taxa de conformidade regimental | % |
| **η** | Aderência às Metas CNJ 1, 2 e 4 | % |

---

## Protocolo experimental

- **Design:** GC (controle) + GE1 (NSGA-II) + GE2 (AMGA2) + GE3 (SPEA2)
- **Replicações:** 30 por grupo → 90 execuções totais
- **Significância:** α = 0,05 com correção de Bonferroni (α_adj ≈ 0,017)
- **Testes:** Shapiro-Wilk → ANOVA ou Kruskal-Wallis → post-hoc Tukey/Dunn

---

## Instalação

```bash
# Clone o repositório
git clone https://github.com/luizcsalmeida/pm4jud.git
cd pm4jud

# Crie um ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Instale as dependências
pip install -r requirements.txt
```

### Dependências principais

```
pm4py>=2.7.0
simpy>=4.0.0
requests>=2.31.0
pandas>=2.0.0
numpy>=1.25.0
scipy>=1.11.0
matplotlib>=3.7.0
tqdm>=4.66.0
lxml>=4.9.0
```

---

## Execução rápida — Fase 1

```bash
# 1. Extração do DATAJUD (requer chave API pública do DPJ/CNJ)
python etl/pm4jud_etl.py --api-key <SUA_CHAVE_CNJ>

# 2. Tratamento de complexidade (D'Castro)
python dcastro/pm4jud_dcastro.py --input output/pm4jud_log_gab_reynaldo.xes

# 3. Mineração de processos
python pm/pm4jud_pm.py --input output/log_filtrado_reynaldo.xes

# 4a. Complementação para LTLf
python complement/pm4jud_complement.py --input output/log_filtrado_reynaldo.xes

# 4b. Verificação declarativa LTLf
python ltlf/pm4jud_ltlf.py --input output/log_complementado_reynaldo.xes

# 5. Simulação DES
python des/pm4jud_des.py --params output/parametros_reynaldo.json

# 6. Otimização (90 execuções)
python opt/pm4jud_opt.py --des output/modelo_des_reynaldo.pkl

# 7. Análise estatística
python stat/pm4jud_stat.py --results output/resultados_90exec.csv
```

---

## Estrutura do repositório

```
pm4jud/
├── etl/                    # P1 — Extração DATAJUD → XES
├── dcastro/                # P2 — Tratamento D'Castro + NLP
├── pm/                     # P3 — Mineração de processos (IMf)
├── complement/             # P4a — Complementação [SIM-LTLf]
├── ltlf/                   # P4b — Verificação declarativa LTLf
├── sim2log/                # PM4JUD-Sim2Log — geração [SIM-DES]
├── des/                    # P5 — Simulação DES (SimPy)
├── opt/                    # P6 — Otimização NSGA-II/AMGA2/SPEA2
├── stat/                   # P7 — Análise estatística
├── viz/                    # PM4JUD-VIZ — dashboards
├── ontology/               # Ontologia PM4JUD — 7 módulos OWL/RDF
├── figuras/                # Scripts de geração das figuras da dissertação
├── output/                 # Saídas geradas (ignorado pelo .gitignore)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Ontologia PM4JUD

A Ontologia PM4JUD é composta por **7 módulos OWL/RDF** desenvolvidos com base no **Modelo Nacional de Interoperabilidade (MNI/CNJ)** e nas **Tabelas Processuais Unificadas (TPU/CNJ)**:

| Módulo | Descrição |
|--------|-----------|
| 1 — Núcleo Estrutural | Entidades MNI 2.2.2: ProcessoJudicial, Parte, Movimento, Documento |
| 2 — Classes Processuais | Tabela de Classes TPU/CNJ (133 classes habilitadas STJ) |
| 3 — Assuntos Processuais | Tabela de Assuntos TPU/CNJ (3.278 assuntos habilitados STJ) |
| 4 — Movimentos Processuais | Tabela de Movimentos TPU/CNJ (616 movimentos habilitados STJ) |
| 5 — Documentos Processuais | Tabela de Documentos TPU/CNJ (1.361 documentos habilitados STJ) |
| 6 — Especialidades | Estratificação Criminal / Cível / Tributário / Previdenciário |
| 7 — Restrições Regimentais e Metas | 5 regras RISTJ + Metas CNJ 1, 2 e 4 |

Os arquivos OWL estão em [`ontology/`](ontology/) e foram desenvolvidos em Protégé 5.6.7 com validação pelo reasoner ELK 0.6.0.

---

## Contexto acadêmico

| | |
|-|-|
| **Programa** | PPGIa — Mestrado em Informática Aplicada |
| **Instituição** | Pontifícia Universidade Católica do Paraná (PUCPR) |
| **Autor** | Luiz Claudio Soares de Almeida |
| **Orientador** | Prof. Dr. Edson Emilio Scalabrin |
| **Período** | 2025–2026 |
| **Framework de referência** | PM4SOS (Ferronato, 2022) |
| **Metodologia** | DSRM — Peffers et al. (2007) |

---

## Citação

Se este trabalho for útil para sua pesquisa, cite:

```bibtex
@mastersthesis{almeida2026pm4jud,
  author    = {Almeida, Luiz Claudio Soares de},
  title     = {{PM4JUD}: Otimização Multiobjetivo com Mineração de Processos
               e Simulação no Contexto do Fluxo Processual em Gabinetes de Magistrado},
  school    = {Pontifícia Universidade Católica do Paraná},
  year      = {2026},
  address   = {Curitiba},
  type      = {Dissertação de Mestrado}
}
```

---

## Referências principais

- FERRONATO, J. J. *PM4SOS: Framework Integrado de Mineração de Processos, Simulação Computacional e Otimização Multiobjetivo para Suporte à Tomada de Decisão Operacional em Centros Cirúrgicos.* Tese (Doutorado em Informática) — PUCPR, Curitiba, 2022.
- PEFFERS, K. et al. A Design Science Research Methodology for Information Systems Research. *Journal of Management Information Systems*, v. 24, n. 3, p. 45–77, 2007.
- VAN DER AALST, W. M. P. *Process Mining: Data Science in Action*. 2. ed. Berlin: Springer, 2016.
- D'CASTRO, R. J. et al. Process Mining Discovery in Judicial Domains. *BRACIS*, 2018.

---

## Licença

Este projeto está licenciado sob a [MIT License](LICENSE).

---

## Contato

**Luiz Claudio Soares de Almeida**  
PPGIa — PUCPR  
📧 soares.claudio@pucpr.edu.br  
🔗 [github.com/luizcsalmeida](https://github.com/luizcsalmeida)

---

> *"Data science approaches tend to be process agnostic whereas process science approaches tend to be model-driven without considering the 'evidence' hidden in the data."* — Wil van der Aalst
