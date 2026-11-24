# EX-3 -- criterio de parada DECLARADO ANTES DA PRIMEIRA EXECUCAO

> Gravado em 2026-08-06, com o repositorio em `e2c5ec1` (branch `docs/ex3-mix-doe`),
> **antes** de gerar o design e **antes** de rodar qualquer lote. Nenhum numero de
> resultado do EX-3 existia quando este arquivo foi escrito; os numeros de
> dimensionamento abaixo vem do piloto ME-1 e dos estudos ja fechados (EX-1, EX-2).
> Regra da campanha (`docs/experimental_campaign.md` sec.8/8.1/8.2): declarar antes,
> rodar uma vez, publicar o que saiu.

## 0. O que o EX-3 estima

Como o ganho pareado da modernizacao (Delta MOS-1 = modernizado - baseline, CRN)
varia com a **composicao da ameaca** `uas_archetype_p_mr` em [0, 1] (fracao de
multirrotor), com a intensidade FIXA no nominal (`lambda_per_h` = 6,0).

O efeito e **MARGINAL**, nao efeito principal de um fatorial: a interacao
lambda x mix continua fora de alcance e permanece escopo do EX-4
(`docs/experimental_campaign.md` sec.2.1). A objecao anti-OFAT registrada em
`docs/analysis_layer.md` sec.3 **NAO e revogada** por este estudo.

## 1. Desenho (fixado antes de rodar)

| Elemento | Valor |
|---|---|
| Fator variado | `uas_archetype_p_mr` em [0, 1] (preset "p_mr" de `_STUDY_FACTORS`) |
| Fator fixo | `lambda_per_h` = **6,0** (nominal; identico ao ponto do EX-1) |
| Amostragem | LHS via `scipy.stats.qmc.LatinHypercube(scramble=True)` (= `create_design`, Spec 026) |
| Pontos de projeto | **20** |
| Escada de replicas | **{96, 192, 384, 768}**, parar no primeiro degrau que satisfaz |
| base_seed (CRN) | **20260807** (NOVA -- disjunta de 20260804/EX-1 e 20260806/EX-2; sec.5) |
| design_seed | 20260807 |
| bootstrap_seed | 20260807 |
| model_seed | 20260807 |
| alpha | 0,05 bicaudal; IC sempre a 95% |

> Com **um** fator, o LHS equivale a **amostragem estratificada** do intervalo: os 20
> pontos caem um em cada estrato de largura 0,05, com posicao sorteada dentro do
> estrato (`scramble=True`). **Nao ha preenchimento multidimensional do espaco a
> reivindicar** -- essa propriedade do LHS so aparece com 2+ fatores (EX-4).
>
> **Consequencia declarada de antemao:** como o sorteio e *dentro* do estrato, os
> extremos 0 e 1 **nao serao amostrados**. O ponto de menor mix caira em [0; 0,05] e
> o de maior em [0,95; 1]. A faixa a declarar no fechamento e a **REALIZADA**, nao a
> nominal (o EX-2 pediu [2; 18] e recebeu [2,99; 17,54]).

### 1.1 Por que 20 pontos (e nao os 16 do EX-2)

- Piso de Loeppky et al. (2009) = 10 x n_fatores = **10**; 20 o dobra. Verificado no
  codigo: `evidence_sufficiency` conta fatores com variancia nao-nula
  (`report.py:742`), entao com 1 fator o piso e 10, nao 20.
- A faixa 12-20 e a orientacao da sec.4 do desenho da campanha; 20 e o topo dela.
- O entregavel operacional do EX-3 e a **fronteira de aplicabilidade** -- o ponto em
  que a curva cruza a margem AC-3 de +0,10. Pelo piloto isso cai em p_mr ~ 0,23;
  estratos de 0,05 resolvem essa regiao melhor que estratos de 0,0625.
- O estrato mais baixo (0; 0,05] fica mais perto do extremo de asa-fixa pura, que e
  onde vive a afirmacao de fronteira.
- 20 pontos com k=5 folds dao 4 pontos por fold (contra 3-4 no EX-2), o que atenua
  -- sem eliminar -- o artefato de `cv.mean.r2` da sec.7.5b.

## 2. Criterio PRIMARIO (o que decide a parada)

> **max sobre os 20 pontos de projeto da semi-largura do IC 95% bootstrap BCa de
> Delta MOS-1 (aor=ALL) <= 0,025 em unidades de MOS-1.**

O `max` e declarado **a priori como estatistica**: nao se escolhe qual ponto olhar
depois de ver os dados.

### 2.1 Ancoragem do 0,025 -- por que ABSOLUTO e por que 1/4 da margem

O criterio e **absoluto porque o limiar de decisao e absoluto**. A margem de
aceitacao do Spec 015 AC-3 (gap MOS-1 >= +0,10) e uma constante; a pergunta que o
EX-3 responde ponto a ponto e *"neste mix, o ganho supera a margem?"*. A precisao
necessaria para responder isso nao depende de quao grande o efeito e -- depende da
distancia entre o efeito e a **margem**.

- **1/4 da margem (0,025)** garante que qualquer ponto cujo efeito verdadeiro esteja
  a >= 0,025 da margem (isto e, >= +0,125 ou <= +0,075) seja classificado com 95% de
  confianca contra ela.
- E o **mesmo numero do EX-1** (que o usou para UM efeito de manchete), agora
  aplicado **do jeito do EX-2** (max sobre pontos). A campanha passa a ter duas
  constantes, ambas ancoradas no AC-3, com regra explicita de qual usar:
  **1/2 da margem quando todos os pontos estao longe dela** (EX-2: menor `ci_low`
  +0,1250, nenhum ponto perto de +0,10) e **1/4 quando a curva a ATRAVESSA** (EX-3:
  o efeito previsto vai de +0,015 a +0,39, cruzando +0,10 em p_mr ~ 0,23).
- Precisao-para-estimacao, e nao poder: e o enquadramento AIPE (*Accuracy In
  Parameter Estimation*, Maxwell, Kelley & Rausch 2008; Kelley & Rausch 2006) --
  dimensiona-se N por largura de IC alvo, nao por poder. Coerente com a licao do
  EX-1 (sec.8.1): com efeito de ~1,5 sigma por replica, significancia ja vem com 20
  replicas; as replicas compram **precisao**.

### 2.2 Por que NAO os outros tres candidatos

**(a) Semi-largura absoluta de 0,050 (o numero do EX-2).** Rejeitada. Ela e
suficiente para um efeito de +0,24, mas no extremo de asa-fixa o efeito previsto e
+0,015 e o IC de +/-0,050 o engoliria mais de tres vezes -- e, pior, deixaria o
ponto **indistinguivel da margem** (`ci_high` ~ +0,065 nao decide nada abaixo de
+0,10 com folga). Manter 0,050 tornaria o criterio satisfeito e o estudo mudo
exatamente onde ele tem mais a dizer.

**(b) Semi-largura RELATIVA ao efeito.** Rejeitada, por tres razoes independentes:

1. **Nao termina.** Um criterio "hw <= r x |efeito|" so tem solucao finita se o
   efeito for nao-nulo, e o custo diverge como 1/efeito^2. Com o efeito previsto de
   +0,0146 em p_mr=0, um alvo de 10% exigiria R ~ 17.000 replicas; 25% exigiria
   ~2.700. Um criterio cujo custo e refem de quao perto de zero a verdade esta nao e
   criterio de parada: e refem.
2. **Gasta esforco onde a decisao ja esta resolvida.** No extremo de asa-fixa a
   resposta de decisao ("fica bem abaixo da margem") ja e clara com precisao
   modesta; e ali que o relativo exige o maximo. Precisao sem valor decisorio.
3. **Circularidade, se calculado sobre o efeito OBSERVADO.** Parar em funcao da
   propria estimativa e *optional stopping* sobre a estatistica de interesse e
   enviesa o efeito publicado. (Se fosse por esse caminho, teria de ser ancorado no
   efeito ESPERADO pelo piloto -- ver (c).)

**(c) HIBRIDO (menos exigente entre piso absoluto e alvo relativo), com o alvo
relativo ancorado no efeito ESPERADO pelo piloto.** Foi a candidata mais forte, e
foi construida e avaliada numericamente antes de ser descartada. A forma testada foi
`T_i = clip(0,25 x Delta_previsto(p_mr_i); 0,025; 0,050)`, que aprovaria em R=96 --
quatro vezes mais barato. Rejeitada por:

1. **Os alvos por ponto dependeriam do coeficiente de p_mr do piloto (0,7826 em
   sigma; 0,3755 no efeito), que e a grandeza MENOS validada de todo o
   dimensionamento** -- o ME-1 rodou **sem** `--config` e a unica calibracao com
   `--config` disponivel (EX-1, EX-2) esta toda em p_mr = 0,5. Um criterio de parada
   nao deve herdar a incerteza que o estudo existe para remover.
2. **Custo nao e restricao nesta campanha** (sec.7.4 do desenho): a unica vantagem
   real do hibrido -- 4x menos replicas -- vale minutos de relogio. Trocar
   auditabilidade por minutos e mau negocio.
3. **Ferramental.** `campaign_postprocess.py` aplica um alvo escalar uniforme; o
   hibrido exigiria alvo por ponto, ou seja, alteracao de ferramental ou um veredito
   calculado por fora que **divergiria do campo `all_points_pass` do proprio
   artefato**. Publicar um artefato cujo veredito interno contradiz o criterio
   declarado e precisamente o tipo de armadilha que esta campanha documenta.

**(d) Criterio em DUAS PARTES (alvo na regiao acima da margem + equivalencia na
regiao abaixo).** Rejeitado **como primario**, adotado **como confirmatorio** (sec.3).
Motivo tecnico decisivo: a parte de equivalencia **nao termina**. Exigir
`ci_high < +0,10` num ponto cujo efeito verdadeiro seja exatamente +0,10 e
insatisfazivel para qualquer R -- nenhuma quantidade de replicas resolve. Um
criterio de parada precisa ter terminacao garantida; semi-largura tem
(hw ~ 1/sqrt(R) -> 0), classificacao contra um limiar nao tem. A leitura de
equivalencia continua sendo o produto mais acionavel do estudo, e por isso e
registrada como confirmatorio e como regra de redacao (sec.9), nao como gatilho.

## 3. Criterios CONFIRMATORIOS (declarados, NAO gatilhantes)

Nenhum destes reprova a parada sozinho; sao registrados junto com o veredito.

1. **Classificacao contra a margem AC-3, por ponto.** Para cada ponto, um de tres
   veredictos: `ci_low > +0,10` (supera a margem), `ci_high < +0,10` (fica aquem da
   margem) ou IC cruzando +0,10 (indeterminado nesta precisao). **Meta declarada:
   nenhum ponto indeterminado, exceto os adjacentes ao cruzamento previsto
   (p_mr ~ 0,23).** E a leitura de equivalencia da candidata (d), rebaixada a
   confirmatorio pelo argumento de terminacao.
2. **Precisao da tendencia.** Semi-largura do IC 95% da variacao total de
   Delta MOS-1 sobre a faixa amostrada (amplitude de p_mr x semi-largura do IC da
   inclinacao OLS sobre os pontos) **<= 0,050**. Mesma ancora do EX-2 (metade da
   margem), aqui aplicada a uma variacao total prevista de ~+0,375 -- ou seja, um
   alvo de ~13% relativo sobre a tendencia.
3. **Q2 do surrogate >= 0,50** (Golbraikh & Tropsha 2002), piso ja versionado em
   `report.py::SURROGATE_Q2_MIN`. **Registrar SEMPRE as duas leituras**: o
   `cv.mean.r2` publicado e o **Q2 poolado** calculado por fora (sec.7.5b do desenho
   da campanha). Previsao: ver sec.8.
4. **Estabilidade entre degraus da escada** (espelha `convergence_stop(0.05, ...)`):
   deriva da media <= 0,25 x semi-largura e variacao relativa de media E desvio < 5%
   entre lotes.

## 4. Dimensionamento -- a base empirica (piloto, nao dado do EX-3)

`sigma(Delta MOS-1)` **CRESCE com o mix**, ao contrario do EX-2, onde caia com
lambda. Com lambda FIXO em 6, o mecanismo 1/sqrt(lambda) do EX-2 desaparece (o
denominador de MOS-1, a contagem de chegadas, e agora constante em distribuicao) e
sobra o termo de mix.

**Ajuste re-derivado nesta sessao**, do zero, sobre o datafarm do ME-1
(`outputs/analysis/zida_shield_me_1/datafarm.parquet`, 32 pontos x 24 replicas),
desconfundindo lambda de mix:

```
log sigma = -1.7141 - 0.4027 * log(lambda) + 0.7826 * p_mr        (R2 = 0.836)
=> a lambda = 6 fixo :  sigma(p_mr) ~ 0.08753 * exp(0.7826 * p_mr)
   p_mr=0.0 -> 0.0875     p_mr=0.5 -> 0.1295     p_mr=1.0 -> 0.1915
```

**Calibracao com `--config` (o ME-1 rodou sem).** Razao medido/piloto:

| Fonte | Ponto | sigma piloto | sigma medido | razao |
|---|---|---|---|---|
| EX-1 (N=200) | lambda=6, p_mr=0,5 | 0,12946 | **0,12578** | 0,972 |
| EX-2 (16 pts) | lambda em [3; 17,5], p_mr=0,5 | -- | -- | **mediana 1,1119** (0,962 a 1,389) |

Adota-se o fator **1,1119** (mediana do EX-2, que e o unico conjunto multi-ponto com
`--config`), replicando a correcao de ~11% que o EX-2 mediu:

```
sigma_previsto(p_mr) = 1.1119 * 0.08753 * exp(0.7826 * p_mr)
                     = 0.0973 (p_mr=0)  ...  0.2129 (p_mr=1)
```

**Checagem da FORMA (feita antes de dimensionar).** O risco central desta previsao e
que o coeficiente 0,7826 nunca foi validado com `--config` (toda a calibracao esta
em p_mr = 0,5). Duas verificacoes:

- correlacao entre `p_mr` e o log do residual do ajuste = **-0,0000** -- a forma
  exponencial nao tem vies sistematico ao longo do eixo do mix;
- forma alternativa (sigma LINEAR em p_mr) da R2 = 0,892 contra 0,899 da
  exponencial, e preve sigma = 0,2035 em p_mr=1 contra 0,2129 -- **4% de diferenca**.
  A previsao nao e sensivel a escolha da forma.

Semi-largura prevista = 1,96 * sigma / sqrt(R):

| R | p_mr=0,00 | p_mr=0,25 | p_mr=0,50 | p_mr=0,75 | p_mr=1,00 |
|---|---|---|---|---|---|
| 96 | 0,0195 | 0,0237 | 0,0288 | 0,0350 | 0,0426 |
| 192 | 0,0138 | 0,0167 | 0,0204 | 0,0248 | 0,0301 |
| **384** | **0,0097** | **0,0118** | **0,0144** | **0,0175** | **0,0213** |
| 768 | 0,0069 | 0,0084 | 0,0102 | 0,0124 | 0,0151 |

R minimo para hw <= 0,025: 59 (p_mr=0), 87 (0,25), 128 (0,50), 189 (0,75),
**279 (p_mr=1)**.

### 4.1 PREVISAO A PRIORI DO DIMENSIONAMENTO (a ser confrontada no fechamento)

> **R=96 reprova (~13 dos 20 pontos, todos com p_mr > 0,32); R=192 reprova (~5
> pontos, p_mr > 0,76); R=384 ACEITA. O ponto que amarra e o de mix MAIS ALTO** --
> invertido em relacao ao EX-2, onde amarrava o de lambda baixo.

**Margem de erro declarada da propria previsao.** Em R=384 o ponto de p_mr~1 fica em
**85% do orcamento** (0,0213 de 0,025). O EX-2 mediu razoes medido/ajustado de 0,96 a
1,39 em torno da mediana -- ou seja, +/-25% em torno do valor calibrado. Se sigma no
extremo de mix alto sair 18% acima do previsto, **R=384 reprova** e a escada vai para
768. Por isso a escada tem quatro degraus declarados, e nao tres: nao se acrescenta
degrau depois de ver o resultado.

**Tambem se preve errar QUAL ponto amarra**, pela licao do EX-2 (sec.4.1 do desenho):
sigma tem ruido proprio e a ordenacao por sigma nao segue exatamente a ordenacao por
p_mr. O que se preve e a **regiao** (mix alto), nao o ponto.

**Custo estimado.** 20 x (96 + 192 + 384) = **13.440 pares** se parar em R=384;
28.800 se for a 768. A ~53 ms/replica (mediana medida na sessao EX-2) isso da ~12 min
de CPU e ~2-3 min de relogio com `--workers 6`. Nota: com lambda fixo em 6 o custo
por replica e ~16 chegadas/run em TODOS os pontos (o EX-1 mediu 15,8), contra a media
de 16,9 do desenho do EX-2 -- comparavel, mas a medida vale so dentro da sessao
(dispersao entre sessoes ~3,3x, sec.7.4). **Custo nao e restricao** -- o
dimensionamento e estatistico.

## 5. base_seed nova (20260807), declarada antes de rodar

EX-1 usou 20260804; EX-2 usou 20260806. Como `_crn_seeds(base, design_point_id, n)`
deriva o banco de `SeedSequence(base).spawn(...)`, reusar qualquer uma delas faria o
**ponto de projeto 0 do EX-3 herdar o banco inteiro** do estudo correspondente, e as
estimativas ficariam correlacionadas sem declaracao.

**Ganho colateral, e aqui ele e maior que no EX-2.** No EX-2 o cruzamento com o EX-1
dependia de o LHS sortear um lambda perto de 6. **No EX-3 lambda e EXATAMENTE 6,0 em
todos os 20 pontos, igual ao EX-1** -- entao o ponto sorteado mais proximo de
p_mr = 0,5 e uma replica quase exata do ponto nominal do EX-1
(+0,1853 [+0,1683, +0,2031]), diferindo **so** pelo banco de seeds. E o cross-check
mais limpo disponivel na campanha, e sera reportado como tal.

## 6. Aninhamento CRN entre degraus da escada

`_crn_seeds(base, dp_id, 384)[:192] == _crn_seeds(base, dp_id, 192)` para **cada**
ponto de projeto, porque `SeedSequence.spawn(n)` usa chaves 0..n-1 (sec.7.2). Logo a
comparacao entre degraus e convergencia sobre a MESMA sequencia, ponto a ponto -- nao
uma segunda amostra independente. **Isto sera verificado em dados**, ponto a ponto e
seed a seed, nao apenas argumentado; se falhar, o protocolo de parada esta invalido e
o script sai com codigo 1.

## 7. Esparsidade e degeneracao -- tratamento declarado

Com lambda fixo em 6,0 **nao se espera replica sem chegada** (o EX-1 mediu `N_eff`
200/200 nesse ponto exato, com 15,8 chegadas/replica). O risco a vigiar aqui e o
**outro extremo**: em p_mr proximo de 0 todos os alvos sao asa-fixa a 130 m/s contra
A-29 a 144 m/s -- tail-chase com fechamento de apenas 14 m/s.

| Resposta | Ausencia significa | Tratamento |
|---|---|---|
| `leak_<motivo>` | zero ocorrencias | **zero-fill** sobre (ponto x replica x config x motivo) |
| `rockets_per_engagement` | replica SEM engajamento | **NAO zero-fill**; media indefinida. Reportar `N_eff` por ponto e por config; parear so onde os dois bracos tem a linha |
| MOS-1 | -- | O agregador emite `0.0` quando `total_arrivals == 0` (aggregator.py:105). O par entra no Delta puxando para zero, sem sinalizar |

**A reportar por ponto de projeto, no minimo:** `N_eff` de `rockets_per_engagement`
por config e a contagem de **replicas degeneradas**.

**Previsao a priori (sec.8):** `N_eff` cheio nos 20 pontos e nos dois bracos, zero
replicas degeneradas. Base: no ME-1 os pontos de mix mais baixo (p_mr = 0,021 / 0,053
/ 0,085) tiveram `N_eff` **24/24 nos dois bracos**, com 0,79-0,82 foguetes por
engajamento no baseline -- ou seja, o tail-chase de asa-fixa **reduz o ganho, mas nao
impede o engajamento**. O unico ponto do ME-1 com `N_eff` 23/24 foi
lambda=2,02 -- escassez de **chegada**, nao de mix, e lambda aqui esta fixo em 6.

## 8. Respostas incluidas e PREVISOES A PRIORI POR RESPOSTA

`MOS-1, MOE-1, MOE-3, MOP-1, MOE-2, MOS-2` -- MOS-1 **primeiro** (`primary =
responses[0]` governa o `decision_summary`/`decision_index`). Confirma a lista do
EX-2, sem ajuste.

- **MOP-2 / MOP-3 excluidas**: constantes de config, variancia exatamente zero. O mix
  nao as move (nem lambda movia).
- **MOS-2**: entra em `--responses` (a tabela mestra a trata pelo ramo
  `config="absolute"`), e recebe bootstrap de **UMA amostra** sobre a media **dentro
  de cada ponto de projeto**. `analysis stats --response MOS-2` levanta `ValueError` e
  NAO sera rodado.

### 8.1 Previsoes (do ajuste linear do ME-1 em (lambda, p_mr), avaliado em lambda=6)

| Resposta | Delta previsto em p_mr=0 | em p_mr=1 | r(p_mr) no ME-1 | r(lambda) medido no EX-2 |
|---|---|---|---|---|
| **MOS-1** | **+0,015** | **+0,390** | +0,918 | -0,852 |
| MOE-1 | +0,022 | +0,263 | +0,918 | -0,79 |
| MOE-3 | +0,024 | +0,462 | +0,978 | -0,47 (k-grupos **nao** rejeitava) |
| **MOE-2** | **+0,027** (ver 8.2) | **-0,334** | **-0,985** | -0,37 (k-grupos p=0,427, **inerte**) |
| MOP-1 | **+645** | **+136** | -0,956 | **+0,91** (sinal OPOSTO) |

**Previsao de k-grupos por tercil:** as **cinco** respostas rejeitam a igualdade
(contra **tres** de cinco no EX-2 -- la MOE-3 dava p=0,071 e MOE-2 p=0,427). Se
MOE-3 ou MOE-2 **nao** rejeitarem aqui, a leitura mecanicista da sec.8.2 esta errada.

**Previsao sobre a manchete:** ao contrario do EX-2, **nem todos os pontos terao o IC
inteiro acima da margem AC-3 de +0,10**. Previsao: os pontos com p_mr < ~0,23 ficam
**abaixo** da margem, e o(s) de p_mr mais baixo pode(m) ter IC cruzando **zero**
(em p_mr=0,03 o efeito previsto e +0,026 contra hw=0,010 em R=384 -- deve sair
significativo, mas por pouco; em R=96 nao sairia). **Isso e resultado, nao problema:
e a fronteira de aplicabilidade da modernizacao.** Regra de redacao na sec.9.

### 8.2 MOE-2: a previsao mais nitida, e por que ela e falsificavel

O mecanismo e forma fechada, nao ajuste (`policies/tracking.py`):

- **modernizado** (Eq.15): `p_track = min(1, L_U/sigma_s)` com `sigma_s = 5` m, no
  anulo (14; 20] km. Asa-fixa `L_U = 14` m -> **1,0**; multirrotor `L_U = 1` m -> **0,2**.
- **baseline** (Eqs.11-13 rev. Spec 016): `p_track = min(1, L_U/(d * theta))` com
  `theta = 2,9088e-4` rad, no anulo (2; 13] km. Asa-fixa satura em **1,0** em todo o
  anulo (a 13 km, 14/3,78 = 3,7 > 1); multirrotor vai de 1,0 (ate ~3,4 km) a 0,26
  (a 13 km).

Logo, em **p_mr = 0 os dois bracos valem 1,0 e Delta MOE-2 = 0 EXATAMENTE**, e o
Delta cresce em modulo com a fracao de multirrotor. Ancorando no EX-1
(p_mr = 0,5 -> Delta MOE-2 = **-0,1467**), a previsao mecanicista e:

> **Delta MOE-2(p_mr) ~ -0,293 * p_mr**, quase linear, passando pela ORIGEM;
> |Pearson| >= 0,95; k-grupos rejeita com p < 0,001.

**O que a falsifica:** se o ponto de menor mix apresentar Delta MOE-2
**significativamente diferente de zero**, a leitura acima esta errada. Note que o
ajuste do ME-1 preve intercepto **+0,027** (nao 0) -- as duas previsoes discordam de
proposito, e o ponto de mix mais baixo decide entre elas. Registrado antes de rodar.

Ressalva permanente (sec.5.2 do desenho): MOE-2 **nao tem ruido proprio** -- e
reportada, nunca usada como evidencia de precisao superior.

### 8.3 Previsao sobre o artefato de Q2 (sec.7.5b)

No EX-2 o `cv.mean.r2` saiu 0,2014 (RF) contra Q2 poolado 0,9138, porque um fold
pegou tres pontos quase na horizontal (sd intra-fold 0,00497). **Previsao para o
EX-3: o artefato NAO se repete com a mesma severidade** -- a resposta varia ~27x aqui
(de +0,015 a +0,39) contra 1,65x no EX-2, entao qualquer fold de 4 pontos sorteados
sobre a faixa tera sd interna grande e denominador saudavel. Previsao concreta:
**`cv.mean.r2` >= 0,5** e, por consequencia, **o gate de evidencia do `report.json`
PASSA pela primeira vez na campanha** (os outros dois motivos ja caem sozinhos:
20 >= 10 pontos, Loeppky; e n_pares >> 34, Cohen). O Q2 poolado sera calculado por
fora de qualquer forma, e as duas leituras publicadas lado a lado.

### 8.4 Previsao sobre custo e modos de falha

- `rockets_per_engagement` (modernizado): **cresce com p_mr** (a penalidade
  dimensional M=0,5 do multirrotor reduz P_hit por tiro, exigindo mais
  re-engajamentos ate o acerto). Baseline ~1,0 e quase plano.
- `leak_pending-evaded`: **cai com p_mr**. Asa-fixa a 130 m/s cruza as 80 nmi
  (148 km) em ~19 min; multirrotor a 29 m/s leva ~85 min. Mais mix multirrotor = mais
  tempo para engajar = menos evasoes.
- `leak_winchester` (modernizado): **fraco/plano em p_mr**. No EX-2 quem o movia era
  lambda, e lambda esta fixo. Baixa confianca -- declarada como tal.
- `leak_winchester` (baseline): **zero em todos os pontos** (o baseline dispara ~1
  foguete por engajamento; nunca esgota os 49).

## 9. Como pontos NAO-SIGNIFICATIVOS serao reportados (decidido antes de ve-los)

Regras de redacao fixadas agora, para que a fronteira de aplicabilidade nao seja
relatada como "prova de efeito nulo".

1. **Afirmacao positiva** ("o ganho neste mix e X [IC]") **somente** se o IC 95% BCa
   excluir zero (equivalente: ASL <= alpha = 0,05).
2. **Se o IC contiver zero**, a unica afirmacao permitida e de **limite**: *"neste
   mix o ganho nao e distinguivel de zero com R replicas; os dados o limitam a no
   maximo `ci_high`"*. **Proibido** escrever "o ganho e zero", "a modernizacao nao
   ajuda" ou "nao ha efeito" -- ausencia de evidencia nao e evidencia de ausencia
   (Altman & Bland 1995; Amrhein, Greenland & McShane 2019).
3. **Independentemente da significancia contra zero**, cada ponto recebe uma
   **classificacao contra a margem AC-3 (+0,10)**, que e a leitura de decisao:
   (i) `ci_low > +0,10` -> **supera a margem**; (ii) `ci_high < +0,10` -> **fica
   aquem da margem** (afirmacao negativa INFORMATIVA, no espirito de equivalencia
   /nao-superioridade); (iii) IC cruzando +0,10 -> **indeterminado nesta precisao**.
4. **A manchete de fronteira** ("abaixo de p_mr ~ X a modernizacao nao supera o
   criterio de aceitacao") so pode ser sustentada por pontos do tipo (ii). Pontos do
   tipo (iii) entram como indeterminados, nunca como apoio.
5. **A localizacao do cruzamento** sera reportada com incerteza propagada da
   inclinacao (+/- hw / inclinacao), nao como um valor pontual.

### 9.1 O `build_decision_index` NAO responde a esta pergunta -- verificado no codigo

`build_decision_index` (`report.py:532`) de fato rebaixa e rotula cenario
nao-significativo, **mas o "cenario" dele e o AoR, nao o ponto de projeto**: ele
filtra a tabela mestra por `response` e ranqueia por `aor`, poolando todos os pontos
de projeto dentro de cada AoR. Em EX-3 cada AoR poolara os 20 mixes e sairaa
largamente significativo. **Conclusao declarada antes de rodar: a leitura automatica
do indice de decisao e citavel para comparar AoRs, e NAO e citavel para a questao de
significancia por nivel de mix.** Esta ultima vem de `by_design_point[]` no
`offpath_*.json`, sob as regras 1-5 acima.

## 10. O que este estudo NAO conclui

- Nada sobre a **interacao** lambda x mix (escopo do EX-4). O efeito e marginal, com
  lambda fixo em 6,0.
- Nada fora da faixa REALIZADA de p_mr (a ser declarada no fechamento; os extremos
  0 e 1 nao serao amostrados).
- A "superficie" do `analysis predict` sera uma **CURVA**: com o mix como unico fator
  variado, o eixo de **lambda** da grade 25x25 fica degenerado (todos os 25 valores
  iguais a 6,0) -- espelho exato da ressalva do EX-2, com os eixos trocados. Ler
  qualquer coisa fora de lambda = 6 nao tem suporte de treino.
