#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
PM4JUD-STAT  v1.0
================================================================================

Dissertação de Mestrado — PPGIa/PUCPR
Título: PM4JUD — Otimização Multiobjetivo com Mineração de Processos e
        Simulação no Contexto do Fluxo Processual em Gabinetes de Magistrado
Autor:  Luiz Claudio Soares de Almeida
Orient: Prof. Dr. Edson Emilio Scalabrin
Ano:    2026

Descrição
---------
P9 do pipeline PM4JUD. Executa a análise estatística do design experimental
controlado (FERRONATO, 2022, Cap. 5), comparando os três algoritmos de
otimização multiobjetivo (NSGA-II, AMGA2, SPEA2) com o Grupo Controle (GC)
para as métricas T̄ (tempo médio de julgamento) e G (coeficiente de Gini)
nos três gabinetes piloto da 3.ª Seção Criminal do STJ.

Alinhamento com PM4SOS (FERRONATO, 2022, Cap. 5)
-------------------------------------------------
O PM4SOS adota Kruskal-Wallis + Wilcoxon com correção de Bonferroni para
comparação de algoritmos evolutivos em design experimental controlado.
O PM4JUD herda integralmente esse protocolo e adiciona:
  • Teste de normalidade Shapiro-Wilk com seleção adaptativa do teste
    paramétrico (ANOVA) ou não-paramétrico (Kruskal-Wallis)
  • Tamanho de efeito: η² (ANOVA) ou r de Rosenthal (Kruskal-Wallis)
  • Geração automática da tabela LaTeX (Apêndice C da dissertação)

Design experimental
-------------------
  Variável independente : algoritmo (NSGA-II, AMGA2, SPEA2)
  Grupos                : GC (grupo controle, sem otimização), GE1 (NSGA-II),
                          GE2 (AMGA2), GE3 (SPEA2)
  Replicações           : 30 por grupo = 120 observações por gabinete
  Análise               : α = 0,05; correção Bonferroni (6 comparações par-a-par)
  α ajustado            : α_adj = 0,05 / 6 ≈ 0,0083

Protocolo estatístico por métrica e gabinete
--------------------------------------------
  1. Shapiro-Wilk (n=30) em cada grupo
     H0: dados provêm de distribuição normal
     Se todos os 4 grupos normais → ANOVA de uma via (paramétrico)
     Se ≥ 1 grupo não-normal      → Kruskal-Wallis (não-paramétrico)

  2. Teste omnibus (α=0,05)
     ANOVA        : scipy.stats.f_oneway
     Kruskal-Wallis: scipy.stats.kruskal

  3. Se omnibus significativo → post-hoc pairwise (6 pares)
     ANOVA  : Tukey HSD ou Bonferroni (t-test bilateral)
     KW     : Mann-Whitney U bilateral (scipy.stats.mannwhitneyu)
     Correção: p × 6 (Bonferroni) — reportar p_adj

  4. Tamanho de efeito
     ANOVA : η² = SS_entre / SS_total
     KW    : r = Z / √N  (r de Rosenthal, onde Z = Φ⁻¹(p_mannwhitney))

  5. Métricas κ e η: se variância = 0 em todos os grupos, skip automático
     com registro "sem variabilidade — teste não aplicável"

Seleção da solução de compromisso (representante do Pareto)
-----------------------------------------------------------
Para cada replicação do P8, a fronteira de Pareto pode conter múltiplas
soluções. O P9 seleciona o representante por compromisso mínimo:
  min ‖f(x) − f*‖₂   onde  f* = (T̄_min, G_min) na fronteira da replicação
Somente soluções viáveis (κ_viol = 0 e η_viol = 0) são consideradas.

Entradas (output/ dos pipelines anteriores)
--------------------------------------------
  p8_relatorio.json        Resultados P8: Pareto por replicação (T̄, G, κ, η)
  des_<gab>.json           Baseline GC: T̄ e G por replicação P7b

Saídas (output/)
----------------
  stat_<gab>_<variavel>.json   Resultado por gabinete × métrica
  stat_relatorio.json          Consolidado final → entrada para §6.5 Parte 2
  apendice_c.tex               Tabela LaTeX para Apêndice C da dissertação

Pipeline completo
-----------------
  P1→P2→P3→P4→P5→P6→P7a→P7b→P8→[P9 STAT]

Referências
-----------
  FERRONATO, J. J. PM4SOS. Tese (Doutorado em Informática) — PUCPR, 2022.
  SHAPIRO, S. S.; WILK, M. B. Biometrika, v.52, n.3-4, pp.591-611, 1965.
  KRUSKAL, W. H.; WALLIS, W. A. JASA, v.47, n.260, pp.583-621, 1952.
  MANN, H. B.; WHITNEY, D. R. Ann. Math. Stat., v.18, n.1, pp.50-60, 1947.
  ROSENTHAL, R. Parametric Measures of Effect Size. In: COOPER, H.; HEDGES,
    L. V. (Eds.). Handbook of Research Synthesis. New York: Russell Sage, 1994.
  BONFERRONI, C. E. Teoria Statistica delle Classi. Firenze, 1936.

Repositório: https://github.com/luizcsalmeida/pm4jud/tree/main/stat
================================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger("PM4JUD-STAT")
log.setLevel(logging.INFO)
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-8s] PM4JUD-STAT — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    log.addHandler(_h)
    log.propagate = False

VERSION = "1.0"


# ===========================================================================
# SEÇÃO 1 — Constantes do protocolo experimental
# ===========================================================================

ALPHA          = 0.05          # nível de significância global
GRUPOS         = ["GC", "nsga2", "amga2", "spea2"]
GRUPOS_LABEL   = {"GC": "GC", "nsga2": "NSGA-II", "amga2": "AMGA2", "spea2": "SPEA2"}
N_PARES        = 6             # C(4,2) comparações par-a-par
ALPHA_ADJ      = ALPHA / N_PARES  # correção Bonferroni ≈ 0.0083

VARIAVEIS = {
    "T_medio": {
        "label":   "T̄ (tempo médio de julgamento, dias norm.)",
        "simbolo": "T_bar",
        "latex":   r"$\bar{T}$",
    },
    "gini": {
        "label":   "G (coeficiente de Gini de distribuição de carga)",
        "simbolo": "G",
        "latex":   r"$G$",
    },
    "kappa": {
        "label":   "κ (taxa de conformidade regimental)",
        "simbolo": "kappa",
        "latex":   r"$\kappa$",
    },
    "eta_cnj": {
        "label":   "η (aderência Metas CNJ 1/2/4)",
        "simbolo": "eta",
        "latex":   r"$\eta$",
    },
}

# Pares de comparação (todos os C(4,2) = 6)
PARES = [
    ("GC", "nsga2"), ("GC", "amga2"), ("GC", "spea2"),
    ("nsga2", "amga2"), ("nsga2", "spea2"), ("amga2", "spea2"),
]


# ===========================================================================
# SEÇÃO 2 — Extração de dados do P8 e P7b
# ===========================================================================

def _representante_pareto(solucoes: List) -> Optional[Tuple[float, float]]:
    """
    Seleciona a solução de compromisso de uma fronteira de Pareto.
    Critério: mínima distância Euclidiana ao ponto ideal f* = (T̄_min, G_min).
    Considera apenas soluções viáveis (κ_viol = 0, η_viol = 0).

    Retorna (T̄, G) da solução de compromisso ou None se nenhuma viável.
    """
    viaveis = [s for s in solucoes if s[1][2] == 0.0 and s[1][3] == 0.0]
    if not viaveis:
        return None
    t_vals = [s[1][0] for s in viaveis]
    g_vals = [s[1][1] for s in viaveis]
    t_ideal = min(t_vals)
    g_ideal = min(g_vals)
    melhor = min(
        viaveis,
        key=lambda s: math.hypot(s[1][0] - t_ideal, s[1][1] - g_ideal)
    )
    return (melhor[1][0], melhor[1][1])


def carregar_p8(input_dir: Path) -> Dict:
    """
    Lê p8_relatorio.json e extrai, por gabinete e algoritmo,
    as séries de 30 valores de T̄ e G (solução de compromisso de cada replicação).
    """
    fpath = input_dir / "p8_relatorio.json"
    if not fpath.exists():
        log.error("p8_relatorio.json não encontrado em %s", input_dir)
        sys.exit(1)

    with open(fpath, encoding="utf-8") as f:
        p8 = json.load(f)

    dados: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for gab_data in p8["gabinetes"]:
        gab = gab_data["gabinete"]
        dados[gab] = {}
        for algo, adata in gab_data["algoritmos"].items():
            t_serie, g_serie = [], []
            for rep in adata["replicacoes"]:
                resultado = _representante_pareto(rep["pareto"])
                if resultado is not None:
                    t_serie.append(resultado[0])
                    g_serie.append(resultado[1])
                else:
                    log.warning("  %s/%s rep%d: sem solução viável na fronteira",
                                gab, algo, rep["replicacao"])
            dados[gab][algo] = {"T_medio": t_serie, "gini": g_serie,
                                 "kappa": [1.0] * len(t_serie),
                                 "eta_cnj": [1.0] * len(t_serie)}
            log.info("  %s / %-8s: %d replicações carregadas",
                     gab, algo.upper(), len(t_serie))
    return dados


def carregar_gc(input_dir: Path, gabinetes: List[str]) -> Dict:
    """
    Lê des_<gab>.json e extrai as séries de 30 valores de T̄ e G do GC
    (Grupo Controle — baseline sem otimização, do P7b).
    """
    gc: Dict[str, Dict[str, List[float]]] = {}
    for gab in gabinetes:
        fpath = input_dir / f"des_{gab}.json"
        if not fpath.exists():
            log.warning("  des_%s.json não encontrado — GC indisponível", gab)
            gc[gab] = {}
            continue
        with open(fpath, encoding="utf-8") as f:
            d = json.load(f)
        t_serie = [r["t_medio_dias"] for r in d["replicacoes"]]
        g_serie = [r["gini"]         for r in d["replicacoes"]]
        k_serie = [r["kappa"]        for r in d["replicacoes"]]
        e_serie = [r["eta"]          for r in d["replicacoes"]]
        gc[gab] = {"T_medio": t_serie, "gini": g_serie,
                   "kappa": k_serie,    "eta_cnj": e_serie}
        log.info("  GC / %-10s: %d replicações carregadas", gab, len(t_serie))
    return gc


# ===========================================================================
# SEÇÃO 3 — Testes estatísticos
# ===========================================================================

def _shapiro(serie: List[float]) -> Tuple[float, float, bool]:
    """Shapiro-Wilk. Retorna (W, p, normal)."""
    if len(set(serie)) == 1:          # variância zero
        return (1.0, 1.0, True)
    W, p = scipy_stats.shapiro(serie)
    return (float(W), float(p), p > ALPHA)


def _eta_quadrado(grupos_vals: List[List[float]]) -> float:
    """Tamanho de efeito η² para ANOVA."""
    grande = [v for g in grupos_vals for v in g]
    media_geral = np.mean(grande)
    ss_total  = sum((v - media_geral) ** 2 for v in grande)
    ss_entre  = sum(
        len(g) * (np.mean(g) - media_geral) ** 2
        for g in grupos_vals
    )
    return float(ss_entre / ss_total) if ss_total > 0 else 0.0


def _r_rosenthal(u_stat: float, n1: int, n2: int) -> float:
    """
    Tamanho de efeito r de Rosenthal a partir do Mann-Whitney U.
    Z ≈ (U - μU) / σU  onde μU = n1*n2/2, σU = √(n1*n2*(n1+n2+1)/12)
    r = Z / √(n1+n2)
    """
    mu_u  = n1 * n2 / 2
    sig_u = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    if sig_u == 0:
        return 0.0
    z = (u_stat - mu_u) / sig_u
    return float(abs(z) / math.sqrt(n1 + n2))


def _interpretar_r(r: float) -> str:
    """Interpretação convencional do r de Rosenthal (Cohen, 1988)."""
    if r >= 0.50: return "grande"
    if r >= 0.30: return "médio"
    if r >= 0.10: return "pequeno"
    return "negligível"


def _interpretar_eta2(eta2: float) -> str:
    if eta2 >= 0.14: return "grande"
    if eta2 >= 0.06: return "médio"
    if eta2 >= 0.01: return "pequeno"
    return "negligível"


def testar_variavel(dados_grupos: Dict[str, List[float]],
                    variavel: str) -> Dict:
    """
    Executa o protocolo estatístico completo para uma variável:
      1. Shapiro-Wilk por grupo
      2. ANOVA ou Kruskal-Wallis (seleção adaptativa)
      3. Post-hoc pairwise com Bonferroni (se omnibus significativo)
      4. Tamanho de efeito

    Parâmetros
    ----------
    dados_grupos : dict {grupo: [30 valores]}
    variavel     : 'T_medio' | 'gini' | 'kappa' | 'eta_cnj'

    Retorna dict com toda a análise.
    """
    grupos_presentes = [g for g in GRUPOS if g in dados_grupos]
    series = [dados_grupos[g] for g in grupos_presentes]

    # Estatísticas descritivas
    descritivas: Dict[str, Dict] = {}
    for g in grupos_presentes:
        v = np.array(dados_grupos[g])
        descritivas[g] = {
            "n":        int(len(v)),
            "media":    float(np.mean(v)),
            "desvpad":  float(np.std(v, ddof=1)),
            "mediana":  float(np.median(v)),
            "ic95_lo":  float(np.mean(v) - scipy_stats.t.ppf(0.975, len(v)-1) * np.std(v, ddof=1) / math.sqrt(len(v))),
            "ic95_hi":  float(np.mean(v) + scipy_stats.t.ppf(0.975, len(v)-1) * np.std(v, ddof=1) / math.sqrt(len(v))),
            "label":    GRUPOS_LABEL.get(g, g),
        }

    # Verificar variância zero (κ e η quando SA=100%)
    todas_iguais = all(np.std(s) == 0 for s in series)
    if todas_iguais:
        log.info("    %s: variância zero em todos os grupos — teste não aplicável",
                 variavel)
        return {
            "variavel":       variavel,
            "descritivas":    descritivas,
            "shapiro":        {},
            "teste_omnibus":  {"tipo": "N/A", "motivo": "variância zero",
                               "stat": None, "p": None, "significativo": False},
            "posthoc":        [],
            "efeito":         {"tipo": "N/A", "valor": None, "interpretacao": "N/A"},
            "conclusao":      "sem variabilidade — todos os grupos idênticos",
        }

    # 1. Shapiro-Wilk
    shapiro_res: Dict[str, Dict] = {}
    todos_normais = True
    for g in grupos_presentes:
        W, p, normal = _shapiro(dados_grupos[g])
        shapiro_res[g] = {"W": round(W, 4), "p": round(p, 4), "normal": normal}
        if not normal:
            todos_normais = False

    # 2. Teste omnibus
    if todos_normais:
        stat, p_omni = scipy_stats.f_oneway(*series)
        tipo_omni = "ANOVA"
        efeito_val = _eta_quadrado(series)
        efeito_tipo = "eta2"
        efeito_interp = _interpretar_eta2(efeito_val)
    else:
        stat, p_omni = scipy_stats.kruskal(*series)
        tipo_omni = "Kruskal-Wallis"
        efeito_val = None   # calculado por par no post-hoc
        efeito_tipo = "r_Rosenthal"
        efeito_interp = "calculado por par"

    significativo = bool(p_omni < ALPHA)
    log.info("    %s | %s: stat=%.4f p=%.4f %s",
             variavel, tipo_omni, stat, p_omni,
             "✓ SIGNIFICATIVO" if significativo else "✗ não significativo")

    # 3. Post-hoc pairwise (apenas se omnibus significativo)
    posthoc: List[Dict] = []
    if significativo:
        for (g1, g2) in PARES:
            if g1 not in dados_grupos or g2 not in dados_grupos:
                continue
            v1 = dados_grupos[g1]
            v2 = dados_grupos[g2]
            if todos_normais:
                # t-test bilateral com Bonferroni
                t_stat, p_raw = scipy_stats.ttest_ind(v1, v2)
                p_adj = min(1.0, float(p_raw) * N_PARES)
                ef_val = None
                ef_tipo = "N/A"
                ef_interp = "N/A"
            else:
                # Mann-Whitney U bilateral
                u_stat, p_raw = scipy_stats.mannwhitneyu(
                    v1, v2, alternative="two-sided")
                p_adj = min(1.0, float(p_raw) * N_PARES)
                ef_val = _r_rosenthal(float(u_stat), len(v1), len(v2))
                ef_tipo = "r_Rosenthal"
                ef_interp = _interpretar_r(ef_val)

            sig_par = p_adj < ALPHA
            posthoc.append({
                "par":            f"{GRUPOS_LABEL.get(g1, g1)} vs {GRUPOS_LABEL.get(g2, g2)}",
                "g1":             g1,
                "g2":             g2,
                "p_raw":          round(float(p_raw), 4),
                "p_adj":          round(p_adj, 4),
                "significativo":  sig_par,
                "efeito_tipo":    ef_tipo,
                "efeito_valor":   round(ef_val, 4) if ef_val is not None else None,
                "efeito_interp":  ef_interp,
            })
            log.info("      %s vs %s: p_adj=%.4f %s (efeito %s)",
                     g1, g2, p_adj,
                     "✓" if sig_par else "✗", ef_interp)

    # Efeito global (ANOVA η² ou N/A para KW — calculado por par)
    efeito_global = {
        "tipo":         efeito_tipo,
        "valor":        round(efeito_val, 4) if efeito_val is not None else None,
        "interpretacao": efeito_interp,
    }

    # Conclusão automática
    n_sig = sum(1 for ph in posthoc if ph["significativo"])
    if not significativo:
        conclusao = (f"Teste {tipo_omni} não significativo (p={p_omni:.4f} > α={ALPHA}). "
                     f"Não há diferença detectável entre os quatro grupos.")
    else:
        pares_sig = [ph["par"] for ph in posthoc if ph["significativo"]]
        conclusao = (f"Teste {tipo_omni} significativo (p={p_omni:.4f} < α={ALPHA}). "
                     f"{n_sig} de {len(posthoc)} pares diferem após Bonferroni "
                     f"(α_adj={ALPHA_ADJ:.4f}): {'; '.join(pares_sig)}.")

    return {
        "variavel":       variavel,
        "descritivas":    descritivas,
        "shapiro":        shapiro_res,
        "todos_normais":  todos_normais,
        "teste_omnibus":  {
            "tipo":          tipo_omni,
            "stat":          round(float(stat), 4),
            "p":             round(float(p_omni), 4),
            "significativo": significativo,
        },
        "posthoc":        posthoc,
        "efeito":         efeito_global,
        "conclusao":      conclusao,
    }


# ===========================================================================
# SEÇÃO 4 — Processamento por gabinete
# ===========================================================================

def processar_gabinete(gabinete: str,
                       dados_p8: Dict,
                       dados_gc: Dict,
                       variaveis: List[str]) -> Dict:
    """
    Executa a análise estatística para um gabinete e as variáveis solicitadas.
    Combina os dados do GC (P7b) com os três tratamentos (P8).
    """
    log.info("=" * 65)
    log.info("Analisando gabinete: %s", gabinete.upper())
    log.info("=" * 65)

    resultados_var: Dict[str, Dict] = {}

    for var in variaveis:
        log.info("  Variável: %s", VARIAVEIS[var]["label"])

        # Montar dict {grupo: série}
        dados_grupos: Dict[str, List[float]] = {}

        # GC (P7b)
        if gabinete in dados_gc and var in dados_gc[gabinete]:
            dados_grupos["GC"] = dados_gc[gabinete][var]

        # Algoritmos (P8)
        if gabinete in dados_p8:
            for algo in ["nsga2", "amga2", "spea2"]:
                if algo in dados_p8[gabinete] and var in dados_p8[gabinete][algo]:
                    dados_grupos[algo] = dados_p8[gabinete][algo][var]

        if len(dados_grupos) < 2:
            log.warning("  %s — %s: dados insuficientes (< 2 grupos)", gabinete, var)
            continue

        resultado = testar_variavel(dados_grupos, var)
        resultados_var[var] = resultado

    return {
        "gabinete":   gabinete,
        "n_grupos":   4,
        "n_rep":      30,
        "alpha":      ALPHA,
        "alpha_adj":  ALPHA_ADJ,
        "n_pares":    N_PARES,
        "resultados": resultados_var,
    }


# ===========================================================================
# SEÇÃO 5 — Geração de LaTeX (Apêndice C)
# ===========================================================================

def _fmt_p(p: Optional[float]) -> str:
    if p is None: return "N/A"
    if p < 0.001: return "$< 0{,}001$"
    return f"${p:.3f}$".replace(".", "{,}")


def _fmt_val(v: Optional[float], dec: int = 4) -> str:
    if v is None: return "---"
    fmt = f"{{:.{dec}f}}"
    return f"${fmt.format(v)}$".replace(".", "{,}")


def gerar_latex_apendice_c(todos_resultados: List[Dict],
                            output_dir: Path) -> None:
    """
    Gera apendice_c.tex com:
      - Tabela de estatísticas descritivas por gabinete × variável × grupo
      - Tabela de testes omnibus (ANOVA / Kruskal-Wallis)
      - Tabela post-hoc pairwise com p ajustados e tamanho de efeito
    """
    lines: List[str] = []
    lines.append("% ===========================================================")
    lines.append("% Apêndice C — Hipóteses Estatísticas e Tabelas de Resultados")
    lines.append("% Gerado automaticamente por PM4JUD-STAT v1.0")
    lines.append(f"% {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("% ===========================================================")
    lines.append("")

    for res_gab in todos_resultados:
        gab = res_gab["gabinete"].capitalize()
        lines.append(f"\\section*{{GAB-{gab}}}")
        lines.append("")

        for var, analise in res_gab["resultados"].items():
            var_info = VARIAVEIS[var]
            lines.append(f"\\subsection*{{{var_info['latex']} — {var_info['label']}}}")
            lines.append("")

            # Tabela descritivas
            lines.append("\\begin{table}[H]")
            lines.append("\\centering")
            lines.append(f"\\caption{{Estatísticas descritivas — {var_info['latex']} — GAB-{gab}}}")
            lines.append("\\begin{tabular}{lcccccc}")
            lines.append("\\hline")
            lines.append("\\textbf{Grupo} & \\textbf{n} & \\textbf{Média} & "
                         "\\textbf{DP} & \\textbf{Mediana} & "
                         "\\textbf{IC 95\\% (inf)} & \\textbf{IC 95\\% (sup)} \\\\")
            lines.append("\\hline")
            for g in GRUPOS:
                if g not in analise["descritivas"]:
                    continue
                d = analise["descritivas"][g]
                label = GRUPOS_LABEL.get(g, g)
                lines.append(
                    f"{label} & {d['n']} & "
                    f"{_fmt_val(d['media'])} & {_fmt_val(d['desvpad'])} & "
                    f"{_fmt_val(d['mediana'])} & "
                    f"{_fmt_val(d['ic95_lo'])} & {_fmt_val(d['ic95_hi'])} \\\\"
                )
            lines.append("\\hline")
            lines.append("\\end{tabular}")
            lines.append("\\end{table}")
            lines.append("")

            # Shapiro-Wilk
            if analise.get("shapiro"):
                lines.append("\\begin{table}[H]")
                lines.append("\\centering")
                lines.append(f"\\caption{{Shapiro-Wilk — {var_info['latex']} — GAB-{gab}}}")
                lines.append("\\begin{tabular}{lccc}")
                lines.append("\\hline")
                lines.append("\\textbf{Grupo} & \\textbf{W} & \\textbf{p} & \\textbf{Normal?} \\\\")
                lines.append("\\hline")
                for g in GRUPOS:
                    if g not in analise["shapiro"]:
                        continue
                    sw = analise["shapiro"][g]
                    label = GRUPOS_LABEL.get(g, g)
                    normal_str = "Sim" if sw["normal"] else "Não"
                    lines.append(
                        f"{label} & ${sw['W']:.4f}$ & {_fmt_p(sw['p'])} & {normal_str} \\\\"
                    )
                lines.append("\\hline")
                lines.append("\\end{tabular}")
                lines.append("\\end{table}")
                lines.append("")

            # Omnibus
            omni = analise["teste_omnibus"]
            sig_str = "sim ($p < \\alpha$)" if omni.get("significativo") else "não ($p \\geq \\alpha$)"
            lines.append(f"\\textbf{{Teste omnibus ({omni['tipo']})}}:")
            if omni["stat"] is not None:
                lines.append(
                    f"stat = ${omni['stat']:.4f}$, $p = {_fmt_p(omni['p'])[1:-1]}$, "
                    f"significativo: {sig_str}."
                )
            else:
                lines.append(f"Não aplicável — {omni.get('motivo','N/A')}.")
            lines.append("")

            # Post-hoc
            if analise.get("posthoc"):
                lines.append("\\begin{table}[H]")
                lines.append("\\centering")
                lines.append(f"\\caption{{Post-hoc pairwise (Bonferroni $\\alpha_{{adj}} = {ALPHA_ADJ:.4f}$) "
                             f"— {var_info['latex']} — GAB-{gab}}}")
                lines.append("\\begin{tabular}{lcccc}")
                lines.append("\\hline")
                lines.append("\\textbf{Par} & \\textbf{p (raw)} & \\textbf{p (adj)} & "
                             "\\textbf{Sig?} & \\textbf{Efeito} \\\\")
                lines.append("\\hline")
                for ph in analise["posthoc"]:
                    sig_ph = "\\checkmark" if ph["significativo"] else "---"
                    ef_str = (f"{ph['efeito_valor']:.3f} ({ph['efeito_interp']})"
                              if ph["efeito_valor"] is not None else "---")
                    lines.append(
                        f"{ph['par']} & {_fmt_p(ph['p_raw'])} & {_fmt_p(ph['p_adj'])} & "
                        f"{sig_ph} & {ef_str} \\\\"
                    )
                lines.append("\\hline")
                lines.append("\\end{tabular}")
                lines.append("\\end{table}")
                lines.append("")

            # Conclusão
            lines.append(f"\\textit{{{analise['conclusao']}}}")
            lines.append("")
            lines.append("\\bigskip")
            lines.append("")

    latex_path = output_dir / "apendice_c.tex"
    latex_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("LaTeX gerado: %s (%d linhas)", latex_path.name, len(lines))


# ===========================================================================
# SEÇÃO 6 — Relatório consolidado
# ===========================================================================

class _NumpyEncoder(json.JSONEncoder):
    """Converte tipos numpy para tipos Python nativos antes de serializar."""
    def default(self, obj):
        if isinstance(obj, np.integer):   return int(obj)
        if isinstance(obj, np.floating):  return float(obj)
        if isinstance(obj, np.bool_):     return bool(obj)
        if isinstance(obj, np.ndarray):   return obj.tolist()
        return super().default(obj)


def _dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, cls=_NumpyEncoder)


def salvar_relatorio(todos_resultados: List[Dict],
                     output_dir: Path,
                     variaveis: List[str]) -> None:
    """Salva stat_relatorio.json com resumo consolidado de todos os gabinetes."""

    # Salvar por gabinete × variável
    for res_gab in todos_resultados:
        gab = res_gab["gabinete"]
        for var, analise in res_gab["resultados"].items():
            fname = output_dir / f"stat_{gab}_{var}.json"
            fname.write_text(
                _dump({**res_gab, "resultados": {var: analise}}),
                encoding="utf-8"
            )

    # Relatório consolidado
    payload = {
        "programa":       "PM4JUD-STAT",
        "versao":         VERSION,
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "descricao":      (
            f"Análise estatística do design experimental PM4JUD. "
            f"4 grupos (GC, NSGA-II, AMGA2, SPEA2) × 30 replicações. "
            f"α = {ALPHA}, α_adj = {ALPHA_ADJ:.4f} (Bonferroni, {N_PARES} comparações)."
        ),
        "protocolo": {
            "alpha":           ALPHA,
            "alpha_bonferroni": ALPHA_ADJ,
            "n_pares":         N_PARES,
            "n_rep":           30,
            "grupos":          list(GRUPOS_LABEL.values()),
            "variaveis":       variaveis,
            "teste_normalidade": "Shapiro-Wilk (α=0.05)",
            "teste_omnibus_normal":    "ANOVA uma via (f_oneway)",
            "teste_omnibus_nao_normal": "Kruskal-Wallis",
            "posthoc_normal":    "t-test bilateral + Bonferroni",
            "posthoc_nao_normal": "Mann-Whitney U bilateral + Bonferroni",
            "efeito_anova":    "η² (eta quadrado)",
            "efeito_kw":       "r de Rosenthal (Z/√N)",
        },
        "proximo_passo": "§6.5 Parte 2 da dissertação — análise e interpretação",
        "gabinetes":     todos_resultados,
    }

    fpath = output_dir / "stat_relatorio.json"
    fpath.write_text(_dump(payload), encoding="utf-8")
    log.info("Relatório: %s", fpath.name)


def imprimir_resumo(todos_resultados: List[Dict]) -> None:
    """Imprime tabela-resumo dos resultados no log."""
    log.info("")
    log.info("=" * 70)
    log.info("RESUMO P9 — ANÁLISE ESTATÍSTICA")
    log.info("=" * 70)
    log.info("  α = %.2f | α_adj = %.4f (Bonferroni, %d comparações)",
             ALPHA, ALPHA_ADJ, N_PARES)
    log.info("")

    for res_gab in todos_resultados:
        gab = res_gab["gabinete"]
        log.info("  GAB-%s:", gab.upper())
        for var, analise in res_gab["resultados"].items():
            omni = analise["teste_omnibus"]
            sig = "✓ SIG" if omni.get("significativo") else "✗ n.s."
            p_str = f"p={omni['p']:.4f}" if omni["p"] is not None else "N/A"
            n_sig_pares = sum(1 for ph in analise.get("posthoc", [])
                              if ph["significativo"])
            log.info("    %-12s | %-16s | %-8s | %s | %d/%d pares sig.",
                     VARIAVEIS[var]["simbolo"],
                     omni["tipo"], p_str, sig,
                     n_sig_pares, len(analise.get("posthoc", [])))
    log.info("")
    log.info("Resultados: stat_relatorio.json | LaTeX: apendice_c.tex")
    log.info("Próximo: §6.5 Parte 2 da dissertação")


# ===========================================================================
# SEÇÃO 7 — Entry point
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pm4jud_stat",
        description="PM4JUD-STAT v1.0 (P9) — Análise estatística do design experimental",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input",    required=True,  type=Path,
                        help="Diretório com p8_relatorio.json e des_<gab>.json")
    parser.add_argument("--output",   required=True,  type=Path,
                        help="Diretório de saída (stat_relatorio.json, apendice_c.tex)")
    parser.add_argument("--gabinetes", nargs="+",
                        default=["reynaldo", "palheiro", "schietti"],
                        help="Gabinetes a analisar (default: todos os 3)")
    parser.add_argument("--variavel", nargs="+",
                        default=list(VARIAVEIS.keys()),
                        choices=list(VARIAVEIS.keys()),
                        help=(
                            "Variável(is) a analisar. "
                            "Opções: T_medio, gini, kappa, eta_cnj "
                            "(default: todas as quatro)"
                        ))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    log.info("PM4JUD-STAT v%s | variáveis: %s | gabinetes: %s",
             VERSION, "+".join(args.variavel), "+".join(args.gabinetes))
    log.info("α = %.2f | α_adj = %.4f | %d comparações pairwise",
             ALPHA, ALPHA_ADJ, N_PARES)

    t0 = time.time()

    # Carregar dados
    log.info("Carregando p8_relatorio.json...")
    dados_p8 = carregar_p8(args.input)

    log.info("Carregando GC (des_<gab>.json)...")
    dados_gc = carregar_gc(args.input, args.gabinetes)

    # Analisar
    todos_resultados = []
    for gab in args.gabinetes:
        res = processar_gabinete(gab, dados_p8, dados_gc, args.variavel)
        todos_resultados.append(res)

    # Salvar
    salvar_relatorio(todos_resultados, args.output, args.variavel)
    gerar_latex_apendice_c(todos_resultados, args.output)

    log.info("Tempo total: %.1fs", time.time() - t0)
    imprimir_resumo(todos_resultados)


if __name__ == "__main__":
    main()
