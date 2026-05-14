# PM4JUD — Process Mining for Judicial Decision-Making

> **Framework integrado de Mineração de Processos, Simulação Computacional e Otimização Multiobjetivo para apoio à tomada de decisão operacional em gabinetes de magistrado do Superior Tribunal de Justiça**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PM4Py](https://img.shields.io/badge/PM4Py-2.7%2B-orange)](https://pm4py.fit.fraunhofer.de/)
[![SimPy](https://img.shields.io/badge/SimPy-4.1%2B-green)](https://simpy.readthedocs.io/)
[![SciPy](https://img.shields.io/badge/SciPy-1.12%2B-blue)](https://scipy.org/)
[![RDFLib](https://img.shields.io/badge/RDFLib-7.0%2B-purple)](https://rdflib.readthedocs.io/)
[![OWL](https://img.shields.io/badge/OWL-2%20DL-darkblue)](https://www.w3.org/TR/owl2-primer/)
[![Protégé](https://img.shields.io/badge/Protégé-5.6.7-red)](https://protege.stanford.edu/)
[![DATAJUD](https://img.shields.io/badge/DATAJUD-CNJ-green)](https://datajud-wiki.cnj.jus.br/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Fase%201%20concluída-yellow)]()
[![PPGIa/PUCPR](https://img.shields.io/badge/PPGIa-PUCPR-darkblue)](https://www.pucpr.br/escola-politecnica/mestrado-doutorado/informatica-aplicada/)

---

## 1. Sobre o projeto

O **PM4JUD** é um framework científico desenvolvido como dissertação
de mestrado no **Programa de Pós-Graduação em Informática Aplicada
(PPGIa)** da **Pontifícia Universidade Católica do Paraná (PUCPR)**,
sob orientação do **Prof. Dr. Edson Emilio Scalabrin**.

O framework adapta o **PM4SOS** (Ferronato, 2022), originalmente
aplicado a centros cirúrgicos hospitalares, ao domínio judicial,
com foco nos gabinetes de ministros da **3.ª Seção do Superior
Tribunal de Justiça (STJ)**, competente para matéria penal e
processual penal. Três disciplinas são integradas para produzir
recomendações de redistribuição processual baseadas em evidências:

- **Mineração de Processos** — descoberta automática do fluxo real de tramitação processual a partir de logs DATAJUD/CNJ
- **Simulação Computacional** — geração e avaliação de cenários alternativos de alocação via modelo DES (SimPy M/M/*c*)
- **Otimização Multiobjetivo** — comparação experimental entre NSGA-II, AMGA2 e SPEA2 sobre quatro variáveis dependentes simultâneas

A **Ontologia PM4JUD** (7 módulos OWL/RDF baseados em MNI/CNJ e
TPU/CNJ) opera como camada semântica transversal a todo o pipeline,
habilitando normalização automática de atividades, verificação
declarativa LTLf e formalização de restrições regimentais sem
acesso aos sistemas privados do STJ na Fase 1.

---

## 2. Problema de pesquisa

> *Como diferentes configurações de otimização multiobjetivo (NSGA-II, AMGA2 e SPEA2) se comportam na redução do tempo de julgamento em gabinetes de magistrados, por meio de experimentação controlada baseada em modelos de simulação gerados a partir de logs de eventos, considerando a especialização por assunto processual, as restrições regimentais do STJ, a priorização legal de determinadas classes de processos e o cumprimento das Metas Nacionais do CNJ?*

---

## 3. Visão geral do pipeline

Nove programas Python formam uma sequência linear dividida em cinco
fases metodológicas. A dependência é estrita: nenhum programa pode ser
executado antes que seu antecessor tenha concluído e gravado os
artefatos na pasta `output/`.

### Artefatos computacionais

| Prog. | Módulo | Descrição |
|-------|--------|-----------|
| P1 | [`etl/pm4jud_etl.py`](etl/pm4jud_etl.py) | Extração DATAJUD/CNJ → Parquet (Etapa 1) e Parquet → XES (Etapa 2); filtro por ministro relator; deduplicação |
| P2 | [`refine/pm4jud_refine1.py`](refine/pm4jud_refine1.py) | Pré-processamento D'Castro Etapa 1: limpeza, canonicalização TPU, threshold MF1; k calibrado por gabinete |
| P3 | [`complement/pm4jud_complement.py`](complement/pm4jud_complement.py) | Injeção de eventos `[SIM-ASSESSOR]` via SPARQL da Ontologia (C1–C16); N=38 assessores, seed=42 — **somente Fase 1** |
| P4 | [`refine/pm4jud_refine2.py`](refine/pm4jud_refine2.py) | Pré-processamento D'Castro Etapa 2: filtro por limiar k (MF1 ≥ 0,75); parâmetros calibrados por gabinete |
| P5 | [`pm/pm4jud_pm.py`](pm/pm4jud_pm.py) | Descoberta IMf por estrato (*cl*); DFG; estimação de parâmetros DES (*e_s · e_o · e_c*); rede de Petri |
| P6 | [`ltlf/pm4jud_ltlf.py`](ltlf/pm4jud_ltlf.py) | Verificação declarativa LTLf/*Declare* — C1–C16 (16 regras); SPARQL Módulo 7; diagnóstico κ · η |
| P7a | [`sim2log/pm4jud_sim2log.py`](sim2log/pm4jud_sim2log.py) | Geração de 30 logs sintéticos `[SIM-DES]` por gabinete a partir de *e_s · e_o · e_c* |
| P7b | [`des/pm4jud_des.py`](des/pm4jud_des.py) | Simulação DES em SimPy (M/M/*c* · prioridade HC); 30 replicações; métricas T̄, G, κ, η do GC |
| P8 | [`opt/pm4jud_opt.py`](opt/pm4jud_opt.py) | 90 execuções: GC + GE1 (NSGA-II) + GE2 (AMGA2) + GE3 (SPEA2); CBR + KNN; fronteiras de Pareto |
| P9 | [`stat/pm4jud_stat.py`](stat/pm4jud_stat.py) | Shapiro-Wilk → ANOVA/Kruskal-Wallis → Bonferroni; exportação LaTeX (Apêndice C) |
| — | [`viz/pm4jud_viz.py`](viz/pm4jud_viz.py) | Dashboard PDF + HTML interativo (Fase 1: DATAJUD · Fase 2: SAGWeb) |

```
P1 ETL → P2 REFINE1 → P3 COMPLEMENT → P4 REFINE2 → P5 PM
                                                        ↓
P9 STAT ← P8 OPT ← P7b DES ← P7a SIM2LOG ← P6 LTLf ←┘
```

| Fase | Descrição | Programas |
|------|-----------|-----------|
| **1 — Preparação de dados** | Extração DATAJUD/CNJ → XES; pré-processamento D'Castro (REFINE1 e REFINE2); complementação de eventos internos `[SIM-ASSESSOR]` via Ontologia (C1–C16) | P1, P2, P3, P4 |
| **2 — Mineração de processos** | Descoberta do modelo AS-IS por estrato (IMf); estimação de parâmetros DES (λ, μ, ρ); verificação declarativa LTLf (C1–C16) | P5, P6 |
| **3 — Simulação computacional** | Geração de logs sintéticos `[SIM-DES]`; modelo DES M/M/*c* com prioridade HC; 30 replicações por configuração | P7a, P7b |
| **4 — Otimização multiobjetivo** | NSGA-II / AMGA2 / SPEA2 + CBR; 90 execuções (GC + GE1 + GE2 + GE3); fronteiras de Pareto (T̄, G, κ, η) | P8 |
| **5 — Análise estatística** | Shapiro-Wilk → ANOVA/Kruskal-Wallis → Bonferroni (α_adj = 0,0083); exportação LaTeX para Apêndice C | P9 |

| Prog. | Módulo | Fase | Entrada | Saída | Descrição |
|-------|--------|------|---------|-------|-----------|
| P1 | `etl/pm4jud_etl.py` | 1 | API DATAJUD/CNJ | `etl_<gab>.xes` | Extração dos movimentos TPU via API pública CNJ |
| P2 | `refine/pm4jud_refine1.py` | 1 | `etl_<gab>.xes` | `refine1_<gab>.xes` | Pré-processamento D'Castro: limpeza e canonicalização TPU |
| P3 | `complement/pm4jud_complement.py` | 1 | `refine1_<gab>.xes` | `complement_<gab>.xes` | Injeção de eventos `[SIM-ASSESSOR]` via SPARQL da Ontologia (C1–C16) |
| P4 | `etl/pm4jud_refine2.py` | 1 | `complement_<gab>.xes` | `refine2_<gab>.xes` | Pré-processamento D'Castro: filtro por limiar k (MF1 ≥ 0,75) |
| P5 | `pm/pm4jud_pm.py` | 2 | `refine2_<gab>.xes` | `e_s/e_o/e_c.*` | Descoberta de modelos (IMf) + estimação de parâmetros DES |
| P6 | `ltlf/pm4jud_ltlf.py` | 2 | `complement_<gab>.xes` + Ontologia | `ltlf_<gab>.json` | Verificação declarativa LTLf (constraints C1–C16) |
| P7a | `sim2log/pm4jud_sim2log.py` | 3 | `e_s/e_o/e_c.*` | `sim2log_<gab>_rep*.xes` | Geração de 30 logs sintéticos `[SIM-DES]` por gabinete |
| P7b | `des/pm4jud_des.py` | 3 | `sim2log_<gab>_rep*.xes` | `des_<gab>.json` | Simulação DES M/M/*c* com prioridade HC: T̄, G, κ, η do GC |
| P8 | `opt/pm4jud_opt.py` | 4 | `des_<gab>.json` | `p8_relatorio.json` | Otimização MOOP (NSGA-II / AMGA2 / SPEA2 + CBR) — 90 execuções |
| P9 | `stat/pm4jud_stat.py` | 5 | `p8_relatorio.json` | `stat_relatorio.json` | Análise estatística: Shapiro-Wilk → ANOVA/KW → Bonferroni |

**Atenção — entrada do P6:** o P6 consome `complement_<gab>.xes`
(saída do P3), não `refine2_<gab>.xes` (P4). Os eventos
`[SIM-ASSESSOR]` injetados pelo P3 são a evidência de conformidade
verificada pelo P6; o filtro k do P4 não os remove do escopo de
verificação.

---

## 4. Indicadores de desempenho

| Símbolo | Indicador | Unidade | Objetivo |
|---------|-----------|---------|---------|
| **T̄** | Tempo médio de julgamento (sojourn interno do gabinete) | Dias reais | Minimizar |
| **G** | Coeficiente de Gini de balanceamento de carga entre assessores | Adimensional [0; 1] | Minimizar |
| **κ** | Taxa de conformidade com os prazos regimentais (RISTJ Arts. 110 e 111) | [0; 1] | Maximizar |
| **η** | Aderência às Metas Nacionais CNJ 1, 2 e 4 | [0; 1] | Maximizar |

---

## 5. Protocolo experimental

- **Design:** GC (controle, sem otimização) + GE1 (NSGA-II) + GE2 (AMGA2) + GE3 (SPEA2)
- **Replicações:** 30 por grupo → **90 execuções totais**
- **Significância:** α = 0,05 · correção de Bonferroni para 6 pares → α_adj = 0,0083
- **Testes:** Shapiro-Wilk → ANOVA ou Kruskal-Wallis → post-hoc Tukey / Dunn
- **Parâmetros P8 (Fase 1):** N = 10, n_gen = 20 (200 avaliações/replicação)
- **Parâmetros P8 (Fase 2):** N = 100, n_gen = 100 (2.000 avaliações/replicação)

---

## 6. Dados — estratégia bifásica

### Fase 1 (atual) — dados públicos DATAJUD/CNJ

- **Fonte:** API pública DATAJUD/CNJ (Res. CNJ nº 331/2020)
- **Escopo:** acervo 2024 dos três gabinetes piloto — Habeas Corpus (TPU 1720)
- **Volume:** 32.031 processos
- **Aprovação ética:** não requerida (dados públicos)
- **Acesso:** chave pública emitida pelo DPJ/CNJ em <https://datajud-wiki.cnj.jus.br/api-publica/>

### Fase 2 (futura) — dados operacionais SAGWeb/STJ

- **Fonte:** SAGWeb/STJ (Sistema de Automação de Gabinetes Web — sistema privado)
- **Conteúdo:** logs das ações internas dos assessores; escaninhos; workflow de documentos
- **Condição:** autorização formal do STJ + aprovação CEP via Plataforma Brasil
- **Status:** em tramitação

Os dados do SAGWeb não são públicos e não estão disponíveis neste
repositório. Na Fase 1, eventos sintéticos `[SIM-ASSESSOR]` preenchem
a ausência de dados internos de assessores; na Fase 2, registros reais
do SAGWeb substituem esses eventos.

---

## 7. Gabinetes piloto

| Gabinete | Ministro | Turma | Processos (2024) | k D'Castro | MF1 |
|----------|----------|-------|-----------------|------------|-----|
| `reynaldo` | Reynaldo Soares da Fonseca | 5.ª T. | 11.395 | 0,20 | 92,1 % |
| `palheiro` | Antonio Saldanha Palheiro | 6.ª T. | 10.148 | 0,30 | 77,5 % |
| `schietti` | Rogerio Schietti Cruz | 6.ª T. | 10.488 | 0,25 | 81,9 % |
| **Total** | | | **32.031** | | |

A escolha de três gabinetes da mesma seção criminal não é arbitrária:
a homogeneidade de matéria processual isola a variável de especialidade
no protocolo experimental, tornando as diferenças de T̄ e G entre
gabinetes atribuíveis à configuração de pessoal e ao volume de entrada,
não a diferenças de domínio jurídico. Habeas Corpus (TPU 1720) é a
classe processual do corpus — sua prioridade regimental preemptiva
determina a modelagem como fila M/M/*c* com prioridade no P7b,
conforme o Art. 94 do RISTJ.

---

## 8. Resultados de referência (Fase 1 · mai/2026)

### Grupo Controle (GC — 30 replicações · IC 95%)

| Gabinete | T̄ GC (dias) | ± σ | G GC | ± σ |
|----------|------------|-----|------|-----|
| Reynaldo | **0,884** | 0,023 | **0,1033** | 0,0133 |
| Palheiro | **0,769** | 0,020 | **0,1006** | 0,0099 |
| Schietti | **0,556** | 0,018 | **0,0922** | 0,0063 |

### Grupos Experimentais (GE1–GE3 · melhor T̄ por replicação)

| Gabinete | T̄ GEs | Redução T̄ | G GEs | Redução G |
|----------|--------|-----------|-------|-----------|
| Reynaldo | 0,838–0,840 | **5,0–5,2 %** | 0,050–0,059 | **43–52 %** |
| Palheiro | 0,720–0,721 | **6,3–6,4 %** | 0,044–0,045 | **55–57 %** |
| Schietti | 0,502       | **9,6–9,8 %** | 0,035–0,036 | **61–62 %** |

### Síntese estatística

| Métrica | Resultado |
|---------|-----------|
| κ e η | 1,000 em **100 %** das soluções Pareto |
| Testes omnibus | p < 0,001 · η² ∈ [0,588; 0,952] nos três gabinetes |
| Equivalência entre algoritmos | p_adj ≥ 0,631 (Bonferroni) |
| Hypervolume médio | 92,44–95,52 % por gabinete × algoritmo |
| Impacto agregado | ~1.590 dias-processo/ano · ~7,2 assessores-equivalentes/ano |

---

## 9. Ontologia PM4JUD

Sete módulos OWL/RDF inter-relacionados via `owl:imports` compõem a
Ontologia PM4JUD v2.0, baseada no Modelo Nacional de
Interoperabilidade (MNI/CNJ) e nas Tabelas Processuais Unificadas
(TPU/CNJ). Esses módulos operam como camada semântica transversal:
todos os programas P1–P9 os consultam para normalização de
atividades, geração de restrições declarativas e verificação de
conformidade com o RISTJ e as Metas CNJ. O `PM4JUD.owl` é o módulo
raiz: contém o Módulo 7 (Restrições Regimentais e Metas CNJ) e
importa os demais seis arquivos.

| Arquivo | Módulo | Conteúdo | SPARQL em runtime |
|---------|--------|----------|-------------------|
| `MNI_Core.owl` | Módulo 1 | Núcleo Estrutural MNI 2.2.2 | — |
| `PM4JUD_Classes.owl` | Módulo 2 | Classes Processuais TPU/CNJ (133 habilitadas STJ) | P1 |
| `PM4JUD_Assuntos.owl` | Módulo 3 | Assuntos TPU/CNJ (3.278 habilitados STJ) | P3 |
| `PM4JUD_Movimentos.owl` | Módulo 4 | Movimentos Processuais TPU (616 habilitados STJ) | P1 |
| `PM4JUD_Documentos.owl` | Módulo 5 | Documentos Processuais TPU (1.361 habilitados STJ) | — |
| `MNI_STJ.owl` | Módulo 6 | Especialidades (Criminal, Cível, Tributário, Previdenciário) | P3 |
| `PM4JUD.owl` | Módulo 7 | Restrições Regimentais (C1–C16) + Metas CNJ 1, 2, 4 — **raiz** | P6, P8 |

**Especificações v2.0 (mai/2026):**
- 39 `NamedIndividual`s; estratégia de *punning* OWL 2 DL nos 4 módulos TPU
- 16 regras axiomáticas (C1–C16) cobrindo Arts. 94, 95, 110, 123, 178 e 179 do RISTJ
- Validação: Protégé 5.6.7 + ELK 0.6.0 (consistência lógica) + SPARQL (5 CQs funcionais)

A DL Query tab do Protégé não pode ser usada para validação porque o
ELK 0.6.0 não suporta `DataHasValue` com `DataPropertyAssertion`. As
As 5 Competency Questions formais (Quadro~3.17 do Cap.~3) e as
verificações técnicas adicionais por módulo devem ser executadas via SPARQL:

```bash
# Abrir PM4JUD.owl no Protégé → SPARQL Query tab → carregar:
sparql/PM4JUD_SPARQL_validacao_v2.sparql
```

---

## 10. Instalação e configuração

```bash
# 1. Clone o repositório
git clone https://github.com/luizcsalmeida/pm4jud.git
cd pm4jud

# 2. Crie e ative o ambiente virtual
python -m venv .venv
source .venv/bin/activate     # Linux/macOS
.venv\Scripts\Activate.ps1   # Windows (PowerShell)

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Configure as variáveis de ambiente
cp .env.example .env
# Edite o .env:
#   DATAJUD_API_KEY=<sua_chave_dpj_cnj>
#   FASE_PM4JUD=1          # 1 = DATAJUD público; 2 = SAGWeb/STJ
#   N_ASSESSORES=38        # Res. STJ N.19/2026
```

### Dependências principais (`requirements.txt`)

```
# =============================================================================
# requirements.txt — PM4JUD Pipeline
# Dissertação de Mestrado — PPGIa/PUCPR
# Autor: Luiz Claudio Soares de Almeida | Ano: 2026
#
# Instalação:
#   pip install -r requirements.txt
#
# Python 3.10+  |  versão de referência pm4py: 2.7.8.3
# =============================================================================

# ── Mineração de processos ────────────────────────────────────────────────────
pm4py>=2.7             # IMf, conformance Declare, XES I/O (P2, P4, P5, P6, P7a)
                       # versão de referência: 2.7.8.3

# ── Simulação computacional ───────────────────────────────────────────────────
simpy>=4.1             # DES M/M/c com prioridade HC (P7b)

# ── Análise estatística ───────────────────────────────────────────────────────
scipy>=1.12            # Shapiro-Wilk, ANOVA, Kruskal-Wallis, Bonferroni (P9)

# ── Manipulação de dados ──────────────────────────────────────────────────────
numpy>=1.26            # operações numéricas (P2–P8)
pandas>=2.2            # DataFrames (P1–P4)

# ── Ontologia e semântica ─────────────────────────────────────────────────────
rdflib>=7.0            # SPARQL sobre Ontologia PM4JUD OWL/RDF (P1, P3, P6)

# ── Parsing e serialização ────────────────────────────────────────────────────
lxml>=5.0              # parsing XML/XES (P1–P2)
pyarrow>=15.0          # leitura/gravação Parquet intermediário (P1)

# ── Pré-processamento de texto ────────────────────────────────────────────────
scikit-learn>=1.3      # TF-IDF e similaridade cosseno — D'Castro (P2, P4)

# ── API e ambiente ────────────────────────────────────────────────────────────
requests>=2.31         # API pública DATAJUD/CNJ (P1)
python-dotenv>=1.0     # variáveis de ambiente (.env) (P1)
tqdm>=4.66             # barras de progresso (P1)

# ── Visualização ──────────────────────────────────────────────────────────────
graphviz>=0.20         # exportação DFG e Petri Nets (P5)

# ── Opcional ──────────────────────────────────────────────────────────────────
# declare4py>=1.1      # verificação LTLf alternativa (P6 usa pm4py nativo)
```

---

## 11. Estrutura de pastas

```
pm4jud/
├── .vscode/
│   └── launch.json    ← configurações VS Code (P1–P9)
│   └── extensions.json← configurações VS Code (P1–P9)
├── etl/               ← P1: extração DATAJUD/CNJ
├── refine/            ← P2 (REFINE1) e P4 (REFINE2)
├── complement/        ← P3: injeção [SIM-ASSESSOR]
├── pm/                ← P5: descoberta de modelos e parâmetros DES
├── ltlf/              ← P6: verificação LTLf declarativa
├── sim2log/           ← P7a: geração de logs sintéticos [SIM-DES]
├── des/               ← P7b: simulação DES (SimPy M/M/c)
├── opt/               ← P8: otimização multiobjetivo
├── stat/              ← P9: análise estatística e exportação LaTeX
├── viz/               ← dashboards e visualizações (opcional)
├── ontologia/
│   ├── MNI_Core.owl              ← Módulo 1: Núcleo Estrutural MNI 2.2.2
│   ├── PM4JUD_Classes.owl        ← Módulo 2: Classes Processuais
│   ├── PM4JUD_Assuntos.owl       ← Módulo 3: Assuntos
│   ├── PM4JUD_Movimentos.owl     ← Módulo 4: Movimentos Processuais
│   ├── PM4JUD_Documentos.owl     ← Módulo 5: Documentos Processuais
│   ├── MNI_STJ.owl               ← Módulo 6: Especialidades
│   └── PM4JUD.owl                ← Módulo 7: Restrições + Metas CNJ (raiz)
├── sparql/
│   └── PM4JUD_SPARQL_validacao_v2.sparql  ← consultas de validação da ontologia
├── output/            ← artefatos gerados
│   └── leiame.txt     ← orientação do link do google drive para acesso aos arquivos dos experimentos
├── .env
├── .env.example
├── LICENSE
├── requirements.txt
├── README.md
├── pyrightconfig.json← configuração dos pacotes das classes python
```

---

## 12. Executando o pipeline

### Via VS Code (recomendado)

1. Abra a pasta `pm4jud/` no VS Code
2. `Ctrl+Shift+P` → **"Python: Select Interpreter"** → selecione `.venv`
3. Abra **Run and Debug** (`Ctrl+Shift+D`) → selecione a configuração → **F5**

O `launch.json` cobre P1 a P9 com variantes de teste rápido para P1
(subconjunto de 100 processos) e P6 (um único gabinete), úteis para
verificar o ambiente antes de processar o corpus completo.

### Via terminal (sequência P1 → P9)

```bash
source .venv/bin/activate   # Linux/macOS

# ── FASE 1 — Preparação do log ──────────────────────────────────────────

# P1 — ETL Etapa 1: extração DATAJUD → Parquet
python etl/pm4jud_etl.py \
  --data-inicio 2024-01-01 --data-fim 2024-12-31 \
  --output-dir output --apenas-etapa 1 --ontologia ontologia

# P1 — ETL Etapa 2: Parquet → XES (executar após Etapa 1)
python etl/pm4jud_etl.py \
  --output-dir output --apenas-etapa 2 --ontologia ontologia

# P2 — REFINE1 (3 gabinetes, k calibrados, threshold=0,80)
python refine/pm4jud_refine1.py \
  --input output --output output --ontologia ontologia

# P3 — COMPLEMENT [SIM-ASSESSOR] (N=38 assessores, seed=42)
FASE_PM4JUD=1 python complement/pm4jud_complement.py \
  --input output --output output \
  --gabinetes reynaldo palheiro schietti \
  --n-assessores 38 --seed 42

# P4 — REFINE2 (3 gabinetes, k calibrados)
python refine/pm4jud_refine2.py \
  --input output --output output --ontologia ontologia

# ── FASE 2 — Mineração de processos ──────────────────────────────────────

# P5 — Descoberta de modelos (IMf) + parâmetros DES
python pm/pm4jud_pm.py \
  --input output --output output --ontologia ontologia \
  --amostra-fitness 2000 --amostra-precisao 500

# P6 — Verificação LTLf (usa complement_<gab>.xes, não refine2)
python ltlf/pm4jud_ltlf.py \
  --input output --output output --ontologia ontologia

# ── FASE 3 — Simulação ───────────────────────────────────────────────────

# P7a — Geração de logs sintéticos [SIM-DES] (30 reps × 1.000 casos)
python sim2log/pm4jud_sim2log.py \
  --input output --output output --ontologia ontologia \
  --n-rep 30 --n-casos 1000

# P7b — Simulação DES: T̄, G, κ, η do Grupo Controle (30 reps)
python des/pm4jud_des.py \
  --input output --output output --ontologia ontologia \
  --n-rep 30

# ── FASE 4 — Otimização ──────────────────────────────────────────────────

# P8 — Otimização MOOP: 3 algoritmos × 30 reps (N=10, n_gen=20 — Fase 1)
python opt/pm4jud_opt.py \
  --input output --output output --ontologia ontologia \
  --n-rep 30 --n-pop 10 --n-gen 20

# ── FASE 5 — Análise estatística ─────────────────────────────────────────

# P9 — Shapiro-Wilk → ANOVA/KW → Bonferroni + exportação LaTeX
python stat/pm4jud_stat.py \
  --input output --output output
```

---

## 13. Saídas por programa

```
output/
├── etl_<gab>.xes               ← P1: log XES bruto + raw_<gab>.parquet
├── refine1_<gab>.xes            ← P2: log pré-processado D'Castro Etapa 1
├── complement_<gab>.xes         ← P3: log com [SIM-ASSESSOR] (8 ev/HC · 6 ev/regular)
├── refine2_<gab>.xes            ← P4: log filtrado por k (MF1 ≥ 0,75)
├── e_s_<gab>.csv                ← P5: parâmetros DES λ, μ, ρ por atividade
├── e_o_<gab>.json               ← P5: modelo organizacional ⟨atividade, recurso⟩
├── e_c_<gab>.json               ← P5: estado corrente (traços incompletos)
├── petri_net_<gab>.pnml         ← P5: rede de Petri (IMf)
├── pm5_relatorio.json           ← P5: MF1, atividades, estratos
├── ltlf_<gab>.json              ← P6: κ, η por trace · violações C1–C16
├── p6_relatorio.json            ← P6: síntese conformidade regimental
├── sim2log_<gab>_rep*.xes       ← P7a: 30 logs [SIM-DES] por gabinete (90 total)
├── des_<gab>.json               ← P7b: T̄, G, κ, η do GC (30 rep · IC 95%)
├── p7b_relatorio.json           ← P7b: síntese 3 gabinetes + configuração DES
├── p8_relatorio.json            ← P8: fronteiras Pareto + HV por gabinete × algoritmo
├── base_casos.json              ← P8: base CBR
├── stat_<gab>_<metrica>.json    ← P9: resultados omnibus + post-hoc
├── stat_relatorio.json          ← P9: síntese completa
└── apendice_c.tex               ← P9: tabelas LaTeX para o Apêndice C
```

Os experimentos da Fase 1 (DATAJUD/CNJ) + Dados sintéticos (SAG-Web) foram disponibilizados em endereço no Google drive em virtude da limitação de 100MB de espaço de armazemanento para os repositórios públicos:
🔗 [Link do Google Drive](https://drive.google.com/drive/folders/1Q5kQRYAK4HyvJBRRmWkmezYM3uOYxH-5?usp=sharing)

---

## 14. Estimativas de tempo

As medições foram realizadas com o corpus completo (32.031 processos,
hardware 2025 — 8 núcleos, 32 GB RAM). Variações de latência da API
DATAJUD são a principal fonte de variabilidade no P1.

| Programa | Tempo médio | Observação |
|----------|------------|------------|
| P1 ETL | 15–30 min | Latência da API DATAJUD/CNJ |
| P2 REFINE1 | 2–5 min | |
| P3 COMPLEMENT | 5–10 min | Inclui raciocínio SPARQL sobre a Ontologia |
| P4 REFINE2 | 3–6 min | |
| P5 PM | 10–20 min | IMf + ajuste de distribuições (exp / lognorm / gamma) |
| P6 LTLf | 12–18 min | ~4–6 min por gabinete; paralelizável |
| P7a SIM2LOG | 2–5 min | |
| P7b DES | 35–55 min | 30 replicações × 3 gabinetes |
| P8 OPT | 90–180 min | Fase 1: N=10, n_gen=20 (200 avaliações/rep) |
| P9 STAT | < 1 min | |
| **Total** | **~3–5 h** | Execução sequencial completa |

Na Fase 2, o P8 operará com N=100, n_gen=100 (2.000 avaliações/rep),
elevando seu tempo para 1.200–2.400 min. A paralelização por gabinete
reduz o tempo total a um terço nessa configuração.

---

## 15. Dados sintéticos — distinção obrigatória

Dois tipos de evento sintético coexistem no pipeline com finalidades
incompatíveis. Cruzar as marcações invalida simultaneamente os resultados
de conformidade (κ) e os de simulação (T̄, G).

| Marcação | Gerado por | Finalidade | Consumido por | Uso proibido |
|----------|-----------|-----------|---------------|-------------|
| `[SIM-ASSESSOR]` | P3 | Representar assessores ausentes no DATAJUD — evidência para LTLf | P5, P6, P8, P9 | Entrada do DES (P7b) |
| `[SIM-DES]` | P7a | Gerar traces completos para calibrar o modelo DES | P7b | Evidência de conformidade (P6) |

O DATAJUD registra apenas movimentos processuais TPU visíveis no
portal público; as ações internas dos assessores ocorrem no SAGWeb
e não são transmitidas ao DATAJUD. O P3 infere e injeta esses
eventos via regras da Ontologia (C1–C16), produzindo `[SIM-ASSESSOR]`.
Na Fase 2, registros reais do SAGWeb substituem esses eventos
sintéticos, e o flag deixa de aparecer nas séries históricas.

---

## 16. Fase 1 vs Fase 2

| Dimensão | Fase 1 (atual) | Fase 2 (futura) |
|----------|---------------|-----------------|
| Fonte | DATAJUD/CNJ (API pública) | SAGWeb/STJ (sistema privado) |
| Assessores | Sintéticos `[SIM-ASSESSOR]` | Reais anonimizados via SAGWeb |
| Aprovação ética | **Não exigida** | CEP via Plataforma Brasil |
| Autorização STJ | **Não exigida** | Autorização formal (em andamento) |
| `FASE_PM4JUD` | `1` | `2` |
| Validade de κ | Consistência interna (Nível 1 real; Nível 2 sintético) | Validação independente |
| Parâmetros P8 | N=10, n_gen=20 | N=100, n_gen=100 |

A distinção entre fases define o escopo de validade das afirmações
sobre conformidade. Na Fase 1, κ=1,000 demonstra consistência
interna do pipeline — resultado esperado por construção. A Fase 2
produzirá a evidência independente necessária para generalizar essa
afirmação a dados reais de assessores.

```bash
# Para executar na Fase 2:
# Configure no .env:
FASE_PM4JUD=2
SAGWEB_DATA_DIR=/caminho/para/dados/sagweb
```

---

## 17. Reprodução dos experimentos

Os datasets anonimizados da Fase 1 e todos os artefatos de configuração
estão disponíveis no repositório. Para reproduzir apenas a análise
estatística a partir dos resultados já gerados pelo P8:

```bash
python stat/pm4jud_stat.py \
  --input output --output output --variavel T_medio gini
```

A validação semântica da Ontologia pode ser executada
independentemente do pipeline. As 5 Competency Questions formais
(CQ1–CQ5, documentadas no Cap.~3) e as verificações técnicas
adicionais por módulo estão disponíveis em:

```bash
# Abrir ontologia/PM4JUD.owl no Protégé 5.6.7
# SPARQL Query tab → carregar:
sparql/PM4JUD_SPARQL_validacao_v2.sparql
```

---

## 18. Problemas comuns

| Erro / Sintoma | Causa provável | Solução |
|----------------|---------------|---------|
| `401 Unauthorized` | Chave API inválida ou expirada | Renovar em <https://datajud-wiki.cnj.jus.br/api-publica/> |
| `RuntimeError: Falha após 5 tentativas` | Instabilidade da API DATAJUD | Aguardar 5–10 min e reexecutar P1 |
| `pm4py not found` | Ambiente virtual não ativado | `source .venv/bin/activate` |
| `rdflib not found` | Dependência ausente | `pip install rdflib` |
| `Resolve missing import?` (Protégé) | `catalog-v001.xml` desatualizado | Repontenciar para `ontologia/PM4JUD_Assuntos.owl` v2.0 |
| `n_crimes_adm_total = 0` (P6) | `PM4JUD_Assuntos.owl` não é v2.0 | Usar versão mai/2026 com `ehCrimeAdministracaoPublica` |
| `TypeError: bool not JSON serializable` (P9) | `numpy.bool_` não convertido | Versão corrigida disponível no repositório |
| XES não gerado (P1) | `pm4py` desatualizado | `pip install --upgrade pm4py` |
| MF1 < 0,75 após P4 | Limiar k muito restritivo | Valores validados: reynaldo=0,20 / palheiro=0,30 / schietti=0,25 |
| `DataHasValue not supported` (ELK) | ELK 0.6.0 não suporta esse axioma | Usar SPARQL — não usar DL Query tab |
| `[SIM-ASSESSOR]` ausente no log P5 | P3 não executado antes do P5 | Verificar a sequência P3 → P4 → P5 |
| Resultados P8 inconsistentes com P7b | `des_<gab>.json` de execução anterior | Reexecutar P7b antes do P8 |

---

## 19. Siglas

| Sigla | Expansão |
|-------|----------|
| AMGA2 | Archive-based Micro Genetic Algorithm 2 |
| API | Application Programming Interface |
| CBR | Case-Based Reasoning — Raciocínio Baseado em Casos |
| CEP | Comitê de Ética em Pesquisa |
| CNJ | Conselho Nacional de Justiça |
| DATAJUD | Base de Dados do Poder Judiciário Nacional |
| DES | Discrete Event Simulation — Simulação por Eventos Discretos |
| DFG | Directly-Follows Graph — Grafo de Sequência Direta |
| DL | Description Logics — Lógicas de Descrição (subconjunto do OWL 2) |
| DPJ | Departamento de Pesquisas Judiciárias (CNJ) |
| DSRM | Design Science Research Methodology |
| GC | Grupo Controle |
| GE | Grupo Experimental |
| HC | Habeas Corpus |
| IC | Intervalo de Confiança |
| IMf | Inductive Miner infrequent — algoritmo de descoberta de processos |
| KNN | K-Nearest Neighbors — K Vizinhos Mais Próximos |
| LTLf | Linear Temporal Logic on Finite traces |
| MF1 | Macro F1-score (métrica de qualidade D'Castro) |
| MNI | Modelo Nacional de Interoperabilidade (CNJ) |
| MOOP | Multi-Objective Optimization Problem — Problema de Otimização Multiobjetivo |
| MTD | Modelo de Transmissão de Dados (DATAJUD) |
| NSGA-II | Non-dominated Sorting Genetic Algorithm II |
| OWL | Web Ontology Language |
| PM | Process Mining — Mineração de Processos |
| PPGIa | Programa de Pós-Graduação em Informática Aplicada |
| PUCPR | Pontifícia Universidade Católica do Paraná |
| RDF | Resource Description Framework |
| RDFS | RDF Schema |
| RISTJ | Regimento Interno do Superior Tribunal de Justiça |
| SAGWeb | Sistema de Automação de Gabinetes Web (STJ) |
| SGT | Sistema de Gestão das Tabelas Processuais Unificadas (CNJ) |
| SPARQL | SPARQL Protocol and RDF Query Language |
| SPEA2 | Strength Pareto Evolutionary Algorithm 2 |
| STJ | Superior Tribunal de Justiça |
| TPU | Tabelas Processuais Unificadas (CNJ) |
| XES | eXtensible Event Stream — padrão IEEE para logs de eventos |

### Variáveis e parâmetros

| Símbolo | Nome | Unidade / Domínio | Contexto |
|---------|------|-------------------|---------|
| **T̄** | Tempo médio de julgamento | Dias reais | Métrica f1 do MOOP — minimizar |
| **G** | Coeficiente de Gini de carga | [0; 1] adimensional | Métrica f2 do MOOP — minimizar |
| **κ** | Taxa de conformidade regimental | [0; 1] | Métrica f3 do MOOP — maximizar |
| **η** | Aderência às Metas CNJ 1, 2 e 4 | [0; 1] | Métrica f4 do MOOP — maximizar |
| **η₁** | Indicador Meta CNJ 1 (processos julgados vs distribuídos) | [0; 1] | Componente de η |
| **η₂** | Indicador Meta CNJ 2 (acervo antigo julgado) | [0; 1] | Componente de η |
| **η₄** | Indicador Meta CNJ 4 (ações prioritárias julgadas) | [0; 1] | Componente de η |
| **λ** | Taxa de chegada de processos | Processos/mês | Parâmetro DES (P7b) |
| **μ** | Tempo mediano de serviço por atividade | Dias | Parâmetro DES — por atividade e gabinete |
| **ρ** | Utilização do sistema | [0; 1] | Parâmetro DES — ρ = λ / (c · μ⁻¹) |
| **c** | Capacidade efetiva de assessores | Assessores ponderados | Parâmetro DES — 24,6 (Fase 1) |
| **k** | Limiar D'Castro (filtro de frequência) | [0; 1] | P4: reynaldo=0,20 / palheiro=0,30 / schietti=0,25 |
| **w** | Peso do assessor por categoria | [0; 1] | P3/P7b: CJ3A=1,0 · CJ2A=0,8 · FC6C=0,7 · FC4IV=0,2 · FC2II=0,1 |
| **N** | Tamanho da população (P8) | Inteiro | Fase 1: N=10 · Fase 2: N=100 |
| **n_gen** | Número de gerações (P8) | Inteiro | Fase 1: 20 · Fase 2: 100 |
| **HV** | Hypervolume da fronteira de Pareto | % | Qualidade das soluções P8 — 92–95 % na Fase 1 |
| **α** | Nível de significância | Probabilidade | α = 0,05 |
| **α_adj** | Nível ajustado (Bonferroni) | Probabilidade | α_adj = 0,05 / 6 ≈ 0,0083 |
| **η²** | Eta-quadrado (tamanho de efeito) | [0; 1] | P9: 0,588–0,952 na Fase 1 |
| **σ** | Desvio-padrão | Mesma unidade da variável | P7b/P9 — 30 replicações |

---

## 20. Contexto acadêmico

| | |
|-|-|
| **Programa** | PPGIa — Mestrado em Informática Aplicada |
| **Instituição** | Pontifícia Universidade Católica do Paraná (PUCPR) |
| **Autor** | Luiz Claudio Soares de Almeida |
| **Orientador** | Prof. Dr. Edson Emilio Scalabrin |
| **Período** | 2025–2026 |
| **Framework de referência** | PM4SOS (Ferronato, 2022) |
| **Metodologia** | DSRM — Peffers et al. (2007) |
| **Contato** | soares.claudio@pucpr.edu.br |

---

## 21. Citação e referências

```bibtex
@mastersthesis{almeida2026pm4jud,
  author  = {Almeida, Luiz Claudio Soares de},
  title   = {{PM4JUD}: Otimiza{\c{c}}{\~a}o Multiobjetivo com
             Minera{\c{c}}{\~a}o de Processos e Simula{\c{c}}{\~a}o
             no Contexto do Fluxo Processual em Gabinetes de Magistrado},
  school  = {Pontif{\'i}cia Universidade Cat{\'o}lica do Paran{\'a} (PUCPR)},
  year    = {2026},
  address = {Curitiba},
  note    = {PPGIa. Orientador: Prof. Dr. Edson Emilio Scalabrin}
}
```

### Referências principais

- FERRONATO, J. J. *PM4SOS: Framework Integrado de Mineração de Processos, Simulação Computacional e Otimização Multiobjetivo para Suporte à Tomada de Decisão Operacional em Centros Cirúrgicos.* Tese (Doutorado em Informática) — PUCPR, Curitiba, 2022.
- PEFFERS, K. et al. A Design Science Research Methodology for Information Systems Research. *Journal of Management Information Systems*, v. 24, n. 3, p. 45–77, 2007.
- VAN DER AALST, W. M. P. *Process Mining: Data Science in Action*. 2. ed. Berlin: Springer, 2016.
- D'CASTRO, R. J. et al. Process Mining Discovery in Judicial Domains. In: *BRACIS*, 2018.
- LENZERINI, M. et al. Metamodeling in OWL 2 QL via Punning. *Artificial Intelligence*, v. 292, p. 103432, 2021.

---

## Licença

Este projeto está licenciado sob a [MIT License](LICENSE).

---

*Última atualização: mai/2026 · Fase 1 concluída · Fase 2 em andamento*

---

## Contato

**Luiz Claudio Soares de Almeida**  
PPGIa — PUCPR  
📧 soares.claudio@pucpr.edu.br  
🔗 [github.com/luizcsalmeida](https://github.com/luizcsalmeida)

---

> *"Data science approaches tend to be process agnostic whereas process science approaches tend to be model-driven without considering the 'evidence' hidden in the data."* — Wil van der Aalst
