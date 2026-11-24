# EX-2 -- criterio de parada DECLARADO ANTES DA PRIMEIRA EXECUCAO

> Gravado em 2026-08-06, com o repositorio em `f11346b` (branch `docs/ex2-lambda-doe`),
> **antes** de gerar o design e **antes** de rodar qualquer lote. Nenhum numero de
> resultado do EX-2 existia quando este arquivo foi escrito. Regra da campanha
> (`docs/experimental_campaign.md` sec.8 e sec.8.1): declarar antes, rodar uma vez,
> publicar o que saiu.

## 0. O que o EX-2 estima

Como o ganho pareado da modernizacao (Delta MOS-1 = modernizado - baseline, CRN)
varia com a pressao de ameaca `lambda_per_h` em [2, 18], com o mix de arquetipos
FIXO no nominal (`uas_archetype_p_mr` = 0.5).

O efeito e **MARGINAL**, nao efeito principal de um fatorial: a interacao
lambda x mix continua fora de alcance e permanece escopo do EX-4
(`docs/experimental_campaign.md` sec.2.1). A objecao anti-OFAT registrada em
`docs/analysis_layer.md` sec.3 NAO e revogada por este estudo.

## 1. Desenho (fixado antes de rodar)

| Elemento | Valor |
|---|---|
| Fator variado | `lambda_per_h` em [2, 18] (preset "lambda" de `_STUDY_FACTORS`) |
| Fator fixo | `uas_archetype_p_mr` = 0.5 |
| Amostragem | LHS via `scipy.stats.qmc.LatinHypercube(scramble=True)` (= `create_design`, Spec 026) |
| Pontos de projeto | **16** |
| Escada de replicas | **{48, 96, 192}**, parar no primeiro degrau que satisfaz |
| base_seed (CRN) | **20260806** (NOVA -- disjunta da 20260804 do EX-1; ver sec.5) |
| design_seed | 20260806 |
| bootstrap_seed | 20260806 |
| model_seed | 20260806 |
| alpha | 0.05 bicaudal; IC sempre a 95% |

> Com **um** fator, o LHS equivale a **amostragem estratificada** do intervalo: os 16
> pontos caem um em cada estrato de largura 1,0 UAS/h, com posicao sorteada dentro do
> estrato (`scramble=True`). Nao ha preenchimento multidimensional do espaco a
> reivindicar -- essa propriedade do LHS so aparece com 2+ fatores (EX-4).

## 2. Criterio PRIMARIO (o que decide a parada)

> **max sobre os 16 pontos de projeto da semi-largura do IC 95% bootstrap BCa de
> Delta MOS-1 (aor=ALL) <= 0,05 em unidades de MOS-1.**

- E o criterio do EX-1 (semi-largura do IC de Delta MOS-1) generalizado para desenho
  multi-ponto. O `max` e declarado **a priori como estatistica**: nao se escolhe qual
  ponto olhar depois de ver os dados.
- **Ancoragem do 0,05:** metade da margem de aceitacao do Spec 015 AC-3 (gap MOS-1
  >= +0,10). O EX-1 usou 1/4 dessa margem para UM numero de manchete; aqui o
  entregavel e uma CURVA, e a exigencia operacional e que **cada ponto da curva
  decida sozinho** se o ganho naquele lambda supera a margem AC-3.
- **O ponto que amarra e o de lambda mais baixo**, porque sigma cai com lambda
  (sec.4) -- e e o mesmo ponto onde a esparsidade da sec.7 morde. O criterio
  amarra deliberadamente o canto mais dificil, nao a media.

## 3. Criterios CONFIRMATORIOS (declarados, NAO gatilhantes)

Nenhum destes reprova a parada sozinho; sao registrados junto com o veredito.

1. **Precisao da tendencia.** Semi-largura do IC 95% da variacao total de
   Delta MOS-1 sobre a faixa amostrada (= amplitude de lambda x semi-largura do IC
   da inclinacao OLS sobre os pontos de projeto) **<= 0,05** -- mesma ancora do
   primario, para que a tendencia fique resolvida com a mesma precisao de um ponto.
2. **Q2 do surrogate >= 0,50** (Golbraikh & Tropsha 2002), o piso ja versionado em
   `report.py::SURROGATE_Q2_MIN` e usado pelo `evidence_sufficiency`.
3. **Estabilidade entre degraus da escada** (espelha `convergence_stop(0.05, ...)`):
   variacao relativa da media E do desvio do Delta agregado < 5% entre lotes, e
   deriva da media <= 0,25 x semi-largura, avaliadas **por ponto de projeto**
   (a versao agregada seria a media do espaco, nao um ponto de operacao).

**Por que nenhum destes e primario.** A inclinacao depende conjuntamente de
`n_pontos` x `replicas`, e um alvo relativo a propria inclinacao seria circular. O Q2
com 16 pontos num k-fold sobre curva 1-D e estatistica ruidosa -- fazer a parada
refem dela seria fragil. Ambos informam a validade, nao o dimensionamento.

## 4. Dimensionamento -- a base empirica (piloto, nao dado do EX-2)

sigma(Delta MOS-1) **nao e constante ao longo de lambda**. Ajuste no datafarm do
estudo ME-1 (`outputs/analysis/zida_shield_me_1/datafarm.parquet`, 32 pontos x 24
replicas), desconfundindo lambda de mix (o ME-1 varia os dois juntos):

```
log sigma = -1.7141 - 0.4027 * log(lambda) + 0.7826 * p_mr        (R2 = 0.836)
=> a p_mr = 0.5 :  sigma(lambda) ~ 0.2653 * lambda^-0.40
   lambda= 2.0 -> 0.2015   lambda= 6.0 -> 0.1295   lambda=18.0 -> 0.0832
```

O expoente ~ -0.4 e **mecanismo, nao ajuste**: MOS-1 e razao cujo denominador e a
contagem de chegadas (proporcional a lambda), logo sigma ~ 1/sqrt(lambda).

**Validacao do modelo-piloto:** em lambda=6, p_mr=0.5 ele preve sigma = 0.1295; o
EX-1 mediu **0.1258** com `--config` (erro de 3%). Isso autoriza usar o piloto para
dimensionar, apesar de o ME-1 ter rodado sem `--config` (fronteira sintetica).

Semi-largura prevista = 1.96 * sigma / sqrt(R):

| R | lambda=2.0 | lambda=6.0 | lambda=18.0 |
|---|---|---|---|
| 48 | 0.0570 | 0.0366 | 0.0235 |
| **96** | **0.0403** | 0.0259 | 0.0166 |
| 192 | 0.0285 | 0.0183 | 0.0117 |

Previsao: **o degrau 48 reprova no ponto de lambda mais baixo e o 96 aprova.** Fica
registrado como previsao a priori, para ser confrontada com o resultado.

**Custo estimado** (0,17 s por replica pareada, mediana medida no `bench_cost.json`
do EX-1): 16x48 = 768 replicas = 131 s de CPU; 16x96 = 1536 = 261 s. Com
`--workers 6` (paralelismo por ponto de projeto, agora efetivo) o relogio cai para
~1 min por lote. **Custo nao e restricao** -- o dimensionamento e estatistico.

## 5. base_seed nova (20260806), declarada antes de rodar

O EX-1 usou `base_seed` 20260804 com `design_point_id = 0`. Como
`_crn_seeds(base, design_point_id, n)` deriva o banco de `SeedSequence(base).spawn(...)`,
reusar 20260804 faria o **ponto 0 do EX-2 herdar o banco inteiro do EX-1**, e as duas
estimativas ficariam correlacionadas sem declaracao.

Com banco disjunto, o ponto do EX-2 mais proximo de lambda=6 vira **replica
independente** da manchete do EX-1 -- cross-check, nao redundancia. O pareamento CRN
que da a potencia e *intra-replica* (modernizado vs baseline compartilham os world
streams) e permanece intacto.

## 6. Aninhamento CRN entre degraus da escada

`_crn_seeds(base, dp_id, 96)[:48] == _crn_seeds(base, dp_id, 48)` para **cada** ponto
de projeto, porque `SeedSequence.spawn(n)` usa chaves 0..n-1 (sec.7.2). Logo a
comparacao 48 vs 96 e convergencia sobre a MESMA sequencia, ponto a ponto -- nao
uma segunda amostra independente. **Isto sera verificado em dados**, ponto a ponto e
seed a seed, nao apenas argumentado; se falhar, o protocolo de parada esta invalido
e o script sai com codigo 1.

## 7. Esparsidade no extremo inferior -- tratamento declarado

Em lambda=2/h a replica pode nao ter engajamento algum. Precedente medido: no ME-1 o
ponto lambda=2,02 teve 23/24 replicas com linha `rockets_per_engagement`.

| Resposta | Ausencia significa | Tratamento |
|---|---|---|
| `leak_<motivo>` | zero ocorrencias | **zero-fill** sobre (ponto x replica x config x motivo) |
| `rockets_per_engagement` | replica SEM engajamento | **NAO zero-fill**; media indefinida. Reportar `N_eff` por ponto e por config; parear so onde os dois bracos tem a linha |
| MOS-1 | -- | O agregador emite `0.0` quando `total_arrivals == 0` (aggregator.py:105). O par entra no Delta puxando para zero |

**A reportar por ponto de projeto, no minimo:** `N_eff` de `rockets_per_engagement`
por config e a contagem de **replicas degeneradas** (Delta MOS-1 = 0 com nenhum
engajamento nos dois bracos -- proxy de replica sem chegada). Este ultimo nao existia
no EX-1 (`N_eff` 200/200) e e a diferenca material do EX-2.

## 8. Respostas incluidas

`MOS-1, MOE-1, MOE-3, MOP-1, MOE-2, MOS-2` -- MOS-1 **primeiro** (`primary =
responses[0]` governa o `decision_summary`/`decision_index`).

- **MOP-2 / MOP-3 excluidas**: constantes de config, variancia exatamente zero;
  lambda nao as move.
- **MOS-2**: entra em `--responses` (a tabela mestra a trata pelo ramo
  `config="absolute"`), e recebe bootstrap de **UMA amostra** sobre a media
  **dentro de cada ponto de projeto** -- nao poolado sobre lambdas, que e exatamente
  o defeito que o `campaign_postprocess.py` corrige.
