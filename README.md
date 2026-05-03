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
               Preparação  Mineração  Simulação  Otimização  Análise   Magistrado
```

| Fase | Descrição | Programas |
|------|-----------|-----------|
| **1 — Preparação de dados** | Extração DATAJUD → XES, classificação por matéria (D'Castro), complementação de eventos internos [SIM-LTLf] ¹ | P1, P2, [P3] |
| **2 — Mineração de processos** | Descoberta do modelo AS-IS por estrato (IMf), parâmetros DES, verificação LTLf (C1–C9) | P4, P5 |
| **3 — Simulação computacional** | Geração de traços sintéticos [SIM-DES], modelo DES M/M/c com prioridade HC, 30 replicações por configuração | Sim2Log, P6 |
| **4 — Otimização multiobjetivo** | CBR + NSGA-II / AMGA2 / SPEA2, 90 execuções (GC + GE1 + GE2 + GE3), fronteira de Pareto | P7 |
| **5 — Análise e visualização** | Shapiro-Wilk, ANOVA/Kruskal-Wallis, Bonferroni α=0,05, dashboards Nível 1 e 2 | P8, VIZ |

> ¹ `[P3]` indica execução condicional: somente no **Nível 1 (DATAJUD/CNJ)**. No Nível 2 (SAGWeb), os eventos internos chegam reais e a complementação sintética é dispensada.

> A **Ontologia PM4JUD** (7 módulos OWL/RDF, baseados em MNI/CNJ e TPU/CNJ) opera como camada semântica transversal a todas as fases, com SPARQL em runtime nos programas P1, P3, P5, P6 e P7.

---

## Artefatos computacionais

| Programa | Módulo | Descrição |
|----------|--------|-----------|
| `P1` | [`etl/pm4jud_etl.py`](etl/pm4jud_etl.py) | Extração DATAJUD → XES; filtro por ministro relator; deduplicação + Parquet |
| `P2` | [`dcastro/pm4jud_dcastro.py`](dcastro/pm4jud_dcastro.py) | Classificação por matéria: 3 perfis D'Castro + TF-IDF; *k* calibrado por gabinete; MF1 ≥ 0,75 |
| `[P3]` | [`complement/pm4jud_complement.py`](complement/pm4jud_complement.py) | Imputação de eventos internos [SIM-LTLf]; escaninhos e workflow de documentos; SPARQL Módulo 6 (C1–C9) — **somente Nível 1** |
| `P4` | [`pm/pm4jud_pm.py`](pm/pm4jud_pm.py) | IMf por estrato (*cl*); DFG; parâmetros *p_v · e_o · e_s · e_c* |
| `P5` | [`ltlf/pm4jud_ltlf.py`](ltlf/pm4jud_ltlf.py) | Verificação Declare — C1–C9 (9 regras); SPARQL Módulo 7; diagnóstico κ · η |
| — | [`sim2log/pm4jud_sim2log.py`](sim2log/pm4jud_sim2log.py) | Geração de traços sintéticos [SIM-DES] a partir de *p_v* + *e_s* |
| `P6` | [`des/pm4jud_des.py`](des/pm4jud_des.py) | Modelo DES em SimPy (M/M/c · prioridade HC); SPARQL Q4 (restrições hard); verificação e validação; 30 replicações por configuração |
| `P7` | [`opt/pm4jud_opt.py`](opt/pm4jud_opt.py) | 90 execuções: GC + GE1 (NSGA-II) + GE2 (AMGA2) + GE3 (SPEA2); CBR · KNN k=5 · TOPSIS |
| `P8` | [`stat/pm4jud_stat.py`](stat/pm4jud_stat.py) | Shapiro-Wilk; ANOVA / Kruskal-Wallis; Bonferroni α=0,05 |
| — | [`viz/pm4jud_viz.py`](viz/pm4jud_viz.py) | Dashboard Nível 1 (DATAJUD) e Nível 2 (SAGWeb) — PDF + HTML interativo |

> ⚠️ **[SIM-LTLf] ≠ [SIM-DES]** — São estratégias de simulação distintas com finalidades, programas e fases diferentes.
> - `[SIM-LTLf]`: completa eventos internos ausentes no DATAJUD (P3 · Fase 1 · somente Nível 1)
> - `[SIM-DES]`: gera cenários hipotéticos para o otimizador (Sim2Log + P6 · Fase 3 · ambos os níveis)
>
> As marcações não devem ser confundidas nem usadas de forma cruzada.

---

## Dados — estratégia bifásica

### Nível 1 (atual) — dados públicos DATAJUD/CNJ

- **Fonte:** API pública DATAJUD/CNJ (Resolução CNJ nº 331/2020)
- **Escopo:** acervo 2024 dos gabinetes piloto (HC · RHC · REsp Criminal)
- **Volume:** 32.031 processos · 728.097 eventos
- **Aprovação ética:** não requerida (dados públicos)
- **Acesso:** chave pública emitida pelo DPJ/CNJ em https://datajud-wiki.cnj.jus.br/api-publica/

### Nível 2 (futuro) — dados operacionais SAGWeb/STJ

- **Fonte:** SAGWeb/STJ (Sistema de Automação de Gabinetes Web)
- **Conteúdo:** logs de ações sistêmicas dos assessores; movimentos internos; escaninhos; workflow de documentos
- **Condição:** autorização formal do STJ + aprovação CEP via Plataforma Brasil
- **Status:** em tramitação

> Os dados do SAGWeb **não são públicos** e não estão disponíveis neste repositório. A estrutura XES é idêntica nos dois níveis; eventos sintéticos [SIM-LTLf] são identificados por `pm4jud:sim_flag = true`, ausente nos eventos reais do Nível 2.

---

## Gabinetes piloto

| Gabinete | Ministro | Turma | Seção | Processos (2024) | *k* D'Castro | MF1 |
|----------|----------|-------|-------|-------------------|--------------|-----|
| GAB-1 | Reynaldo Soares da Fonseca | 5.ª Turma | 3.ª Seção (Criminal) | 11.395 | 0,20 | 92,1% |
| GAB-2 | Sebastião Reis Júnior (Palheiro) | 6.ª Turma | 3.ª Seção (Criminal) | 10.148 | 0,30 | 77,5% |
| GAB-3 | Rogerio Schietti Cruz | 6.ª Turma | 3.ª Seção (Criminal) | 10.488 | 0,25 | 81,9% |
| **Total** | | | | **32.031** | | |

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

- **Design:** GC (controle · sem otimização) + GE1 (NSGA-II) + GE2 (AMGA2) + GE3 (SPEA2)
- **Replicações:** 30 por grupo → **90 execuções totais**
- **Significância:** α = 0,05 com correção de Bonferroni (α_adj ≈ 0,017)
- **Testes:** Shapiro-Wilk → ANOVA ou Kruskal-Wallis → post-hoc Tukey / Dunn

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

## Execução — Nível 1 (DATAJUD/CNJ)

```bash
# P1 — Extração DATAJUD (requer chave API pública do DPJ/CNJ)
python etl/pm4jud_etl.py --api-key <SUA_CHAVE_CNJ> --gabinete reynaldo

# P2 — Classificação por matéria (D'Castro + TF-IDF)
python dcastro/pm4jud_dcastro.py --input output/pm4jud_log_reynaldo.xes

# [P3] — Complementação de eventos internos [SIM-LTLf]  ← somente Nível 1
python complement/pm4jud_complement.py --input output/log_dcastro_reynaldo.xes

# P4 — Mineração de processos (IMf por estrato)
python pm/pm4jud_pm.py --input output/log_complement_reynaldo.xes

# P5 — Verificação LTLf (C1–C9)
python ltlf/pm4jud_ltlf.py --input output/log_complement_reynaldo.xes

# Sim2Log — Geração de traços sintéticos [SIM-DES]
python sim2log/pm4jud_sim2log.py --params output/parametros_reynaldo.json

# P6 — Simulação DES (30 replicações por configuração)
python des/pm4jud_des.py --params output/parametros_reynaldo.json

# P7 — Otimização multiobjetivo (90 execuções: GC + GE1 + GE2 + GE3)
python opt/pm4jud_opt.py --des output/modelo_des_reynaldo.pkl

# P8 — Análise estatística (Shapiro-Wilk + ANOVA/KW + Bonferroni)
python stat/pm4jud_stat.py --results output/resultados_90exec.csv

# VIZ — Dashboard PDF + HTML interativo
python viz/pm4jud_viz.py --results output/resultados_90exec.csv
```

---

## Estrutura do repositório

```
pm4jud/
├── etl/                    # P1 — Extração DATAJUD → XES
├── dcastro/                # P2 — Classificação D'Castro + TF-IDF
├── complement/             # [P3] — Complementação [SIM-LTLf] (Nível 1)
├── pm/                     # P4 — Mineração de processos (IMf)
├── ltlf/                   # P5 — Verificação declarativa LTLf (C1–C9)
├── sim2log/                # Sim2Log — Geração de traços [SIM-DES]
├── des/                    # P6 — Simulação DES (SimPy)
├── opt/                    # P7 — Otimização NSGA-II / AMGA2 / SPEA2
├── stat/                   # P8 — Análise estatística
├── viz/                    # VIZ — Dashboards PDF + HTML
├── ontology/               # Ontologia PM4JUD — 7 módulos OWL/RDF
├── figuras/                # Scripts de geração das figuras da dissertação
├── output/                 # Saídas geradas (ignorado pelo .gitignore)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Ontologia PM4JUD

A Ontologia PM4JUD é composta por **7 módulos OWL/RDF** desenvolvidos em **Protégé 5.6.7** (reasoner ELK 0.6.0) com base no **Modelo Nacional de Interoperabilidade (MNI/CNJ)** e nas **Tabelas Processuais Unificadas (TPU/CNJ)**. A estratégia de *punning* OWL 2 DL (Lenzerini et al., 2021) permite que cada entrada TPU opere simultaneamente como classe e como indivíduo.

| Módulo | Descrição | SPARQL em runtime |
|--------|-----------|-------------------|
| 1 — Núcleo Estrutural | Entidades MNI 2.2.2: ProcessoJudicial, Parte, Movimento, Documento | — |
| 2 — Classes Processuais | Tabela de Classes TPU/CNJ (133 classes habilitadas STJ) | — |
| 3 — Assuntos Processuais | Tabela de Assuntos TPU/CNJ (3.278 assuntos habilitados STJ) | — |
| 4 — Movimentos Processuais | Tabela de Movimentos TPU/CNJ (616 movimentos habilitados STJ) | P1 |
| 5 — Documentos Processuais | Tabela de Documentos TPU/CNJ (1.361 documentos habilitados STJ) | — |
| 6 — Especialidades | Estratificação Criminal / Cível / Tributário / Previdenciário; regras C1–C9 | P3 |
| 7 — Restrições Regimentais e Metas | 5 regras RISTJ (arts. 91 · 177 · 202 · 203 · 34); Metas CNJ 1, 2 e 4 | P5 · P6 · P7 |

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
- LENZERINI, M. et al. Metamodeling in OWL 2 QL via Punning. *Artificial Intelligence*, v. 292, p. 103432, 2021.

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
