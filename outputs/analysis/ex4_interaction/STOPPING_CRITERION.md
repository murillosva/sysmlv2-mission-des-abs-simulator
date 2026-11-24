# EX-4 -- criterio de parada DECLARADO ANTES DA PRIMEIRA EXECUCAO

> **Nota de publicacao (2026-11-20).** Unicas alteracoes editoriais: este aviso e duas
> referencias (na tabela de configuracao e na secao final) que citavam um documento interno
> do desenvolvimento: a da tabela agora cita `docs/reproducibility.md` e a da secao final
> foi retirada. O criterio,
> os alvos, a escada e as previsoes nao mudaram. A versao registrada antes do primeiro lote
> e o blob `02cc5d7e2ff5370cfa9f1e11f20320b235a403fc`, do commit `f076450` do repositorio de
> desenvolvimento (que nao e publico); estas alteracoes mudam o hash deste arquivo.
> (2026-11-23) O estudo-piloto ME-1 citado aqui nao faz parte desta publicacao.

> Gravado em 2026-08-07, com o repositorio em `23cea10` (branch
> `docs/ex4-interaction`), **antes** de gerar o design e **antes** de rodar
> qualquer lote. Nenhum numero de resultado do EX-4 existia quando este arquivo
> foi escrito; todo numero de dimensionamento abaixo vem do piloto ME-1 e dos
> tres estudos ja fechados (EX-1, EX-2, EX-3). Regra da campanha
> (`docs/experimental_campaign.md` sec.8/8.1/8.2/8.3): declarar antes, rodar uma
> vez, publicar o que saiu.
>
> Commitado em commit proprio ANTES do primeiro lote, como o EX-3 fez em
> `18b0446` -- o historico do git prova a anterioridade em vez de este texto
> apenas afirma-la.

## 0. O que o EX-4 estima

Como o ganho pareado da modernizacao (Delta MOS-1 = modernizado - baseline, CRN)
varia com a **intensidade** (`lambda_per_h` em [2, 18]) e a **composicao**
(`uas_archetype_p_mr` em [0, 1]) da ameaca **variadas JUNTAS**, e em particular
qual e o **termo de interacao lambda x mix**.

Este e o primeiro estudo da campanha cujo efeito **NAO e marginal**: sao efeitos
principais **mais** a interacao. **A objecao anti-OFAT registrada em
`docs/analysis_layer.md` sec.3 e em `docs/experimental_campaign.md` sec.2.1 e,
com este estudo, RESPONDIDA** -- nao contornada. O arco aberto no EX-0 fecha
aqui.

## 1. Desenho (fixado antes de rodar)

| Elemento | Valor |
|---|---|
| Fatores variados | **AMBOS**: `lambda_per_h` em [2, 18] e `uas_archetype_p_mr` em [0, 1] (presets "lambda" e "p_mr" de `_STUDY_FACTORS`) |
| Fatores fixos | **nenhum** (sem `--fixed`) |
| Amostragem | **LHS 2-D** via `scipy.stats.qmc.LatinHypercube(d=2, scramble=True)` (= `create_design`, Spec 026) |
| Pontos de projeto | **32** |
| Escada de replicas | **{192, 384, 768, 1536}**, parar no primeiro degrau que satisfaz |
| base_seed (CRN) | **20260808** (NOVA -- sec.5) |
| design_seed | 20260808 |
| bootstrap_seed | 20260808 |
| model_seed | 20260808 |
| alpha | 0,05 bicaudal; IC sempre a 95% |
| config base | `configs/zida_shield_default.yaml` + `sysml/` (**obrigatorio**, docs/reproducibility.md) |
| max_arrivals | OMITIDO (a janela do Spec 018 encerra o run; capar achataria lambda) |
| workers | 6 |

> **Aqui, e so aqui, a propriedade de preenchimento multidimensional do espaco
> se aplica.** Nos EX-2 e EX-3 o LHS 1-D era **amostragem estratificada** de um
> intervalo, e os `STOPPING_CRITERION.md` daqueles estudos dizem isso
> explicitamente. Com 2 fatores, o LHS passa a garantir que a projecao sobre
> **cada** eixo cubra os 32 estratos -- a propriedade pela qual o metodo e
> citado (McKay, Beckman & Conover 1979). Esta redacao pode e deve reivindica-la.
>
> **Consequencia declarada de antemao:** o sorteio continua sendo *dentro* do
> estrato, entao os extremos 2 / 18 e 0 / 1 **nao serao amostrados**. A faixa a
> declarar no fechamento e a **REALIZADA**. Com 32 estratos, o de lambda tem
> largura 0,5 (o mais baixo e [2,0 ; 2,5]) e o de p_mr largura 0,03125.

### 1.1 O ARGUMENTO ESTRUTURAL -- por que o EX-4 e obrigatorio (verificado)

A uniao de **todos** os pontos de projeto ja rodados (EX-1: 1 ponto; EX-2: 16;
EX-3: 20 = **37 pontos**) forma um desenho em **CRUZ**: um braco varia lambda
com p_mr = 0,5, o outro varia p_mr com lambda = 6,0. Verificado nesta sessao
sobre os artefatos publicados (`offpath_n200.json`, `offpath_r96.json`,
`offpath_r192.json`):

```
matriz [1, lambda, p_mr, lambda*p_mr] nos 37 pontos:
  valores singulares = [5.834e+01, 7.001e+00, 2.824e+00, 1.278e-15]
  posto = 3  (nao 4)        numero de condicao = 4.56e+16
  vetor no nucleo (-1/2, 1/12, 1, -1/6):  max |X v| = 1.11e-16
```

A quarta coluna e combinacao linear **EXATA** das outras tres. Prova elementar:
num ponto `(L, 0.5)` vale `-1/2 + L/12 + 1/2 - (1/6)(0.5 L) = 0`; num ponto
`(6, p)` vale `-1/2 + 1/2 + p - (1/6)(6 p) = 0`. Portanto:

> **O coeficiente de interacao NAO tem grau de liberdade nos dados existentes.**
> Nenhuma reanalise, bootstrap ou surrogate sobre o que ja foi rodado o
> recupera. So um desenho com pontos FORA das duas retas.

Para contraste, um LHS 2-D de 32 pontos da **posto 4 com numero de condicao
122** -- 14 ordens de grandeza melhor condicionado. Esta e a justificativa de
metodo do EX-4, e ela e estrutural, nao uma questao de precisao.

### 1.2 Por que 32 pontos (e nao 20, nem 48)

- **Piso de Loeppky et al. (2009) = 10 x n_fatores = 20.** Verificado no codigo:
  `build_consolidated_report` conta fatores com variancia nao-nula
  (`report.py:738-742`), entao com **2** fatores o piso **dobra** para 20 --
  contra 10 nos estudos 1-D. 32 o supera com 60% de folga. O EX-3 foi o primeiro
  estudo a passar nesse gate; **nao se regride nisso**.
- **32 e o numero registrado em `docs/experimental_campaign.md` sec.8.4**, e o
  mesmo do piloto ME-1 (comparabilidade direta do ajuste de interacao).
- **6-7 pontos por fold** com k=5, contra 4 no EX-3 e 3-4 no EX-2 -- atenua o
  artefato de `cv.mean.r2` da sec.7.5b.
- **Detectabilidade da interacao (sec.4.3):** com 32 pontos a semi-largura
  projetada do IC de `b3` e ~0,0061, contra a magnitude de 0,014776 medida no
  piloto -- razao 2,4, ou `|t| ~ 4,8`. Detectavel com folga.
- **Por que NAO 48:** ganharia 20% na precisao de `b3` (0,0061 -> 0,0049) e
  **pioraria o criterio primario** (estratos mais finos alcancam cantos piores,
  e ha mais sorteios para o `max` percorrer). Nao e troca decisiva, e afastaria
  o estudo do numero pre-registrado sem razao forte.

## 2. Criterio PRIMARIO (o UNICO que decide a parada)

> **max sobre os 32 pontos de projeto da semi-largura do IC 95% bootstrap BCa de
> Delta MOS-1 (aor=ALL) <= 0,025 em unidades de MOS-1.**

O `max` e declarado **a priori como estatistica**: nao se escolhe qual ponto
olhar depois de ver os dados.

### 2.1 Ancoragem -- 1/4 da margem, pela regra ja fixada na campanha

A regra da sec.4.2 do desenho da campanha: **1/2 da margem AC-3 (0,050) quando
todos os pontos estao longe dela; 1/4 (0,025) quando a superficie a ATRAVESSA.**
Aqui a superficie atravessa: a reconstrucao bilinear consistente das marginais
medidas (sec.4.4) preve a fronteira `Delta = +0,10` caindo **dentro** do espaco
amostrado em todo o eixo de lambda (p* de ~0,22 a ~0,36). Logo, **cada ponto
precisa ser CLASSIFICADO contra a margem, nao apenas resolvido** -- e o alvo e
0,025, o mesmo do EX-1 e do EX-3.

O limiar vem do **limiar de decisao** (a margem AC-3, uma constante do Spec 015),
**nao do tamanho do efeito**. Precisao-para-estimacao (AIPE; Maxwell, Kelley &
Rausch 2008), nao poder.

### 2.2 O SEGUNDO PRIMARIO SUGERIDO E REJEITADO -- e a razao e medida, nao retorica

`docs/experimental_campaign.md` sec.8.4 registrou como sugestao um **segundo
criterio primario**: *"semi-largura do IC do coeficiente de interacao <= 25% da
magnitude prevista pelo piloto (<= 0,0037)"*. **Rejeitado como primario**, por
duas razoes independentes -- a primeira delas quantificada nesta sessao, antes
de rodar.

**(a) NAO TERMINA na escada. Este e o argumento decisivo.** A incerteza de `b3`
tem duas fontes: o ruido amostral das medias por ponto (que cai com `1/sqrt(R)`)
e a **falta de ajuste** -- a curvatura da superficie verdadeira que um modelo
bilinear nao captura, que **nao cai com R nenhum**. Medido no ME-1 (32 pontos,
R=24, unico conjunto 2-D):

```
sd residual em torno da superficie bilinear   = 0.03248
ruido amostral de uma media por ponto (R=24)  = 0.02459
=> componente de FALTA DE AJUSTE               = 0.02122     (nao encolhe com R)
```

Projetando para o EX-4 (32 pontos, sigma da superficie ancorada da sec.4):

| R | semi-largura projetada de `b3` |
|---|---|
| 192 | 0,00628 |
| 384 | 0,00609 |
| 768 | 0,00592 |
| **3072** | **0,00584** |

**Multiplicar as replicas por 16 compra 7% de precisao.** O alvo de 0,0037 e
**inatingivel com 32 pontos para qualquer R**; alcanca-lo exigiria ~96 pontos de
projeto. Um criterio que a escada nao consegue satisfazer *subindo a escada* e
um criterio nao-terminante -- exatamente o defeito que desqualificou o criterio
**relativo** no EX-3 (sec.4.2 do desenho da campanha), aqui reaparecendo por um
mecanismo diferente (saturacao por falta de ajuste, em vez de divergencia
`1/efeito^2`).

**(b) Herda a incerteza que o estudo existe para remover.** O alvo esta ancorado
na magnitude prevista **pelo piloto** -- e o ME-1 rodou **sem `--config`**, com
fronteira sintetica e sem o bloco `arc` do Spec 030. Foi precisamente esse o
argumento que derrubou o criterio hibrido no EX-3, cujo coeficiente de piloto se
revelou **47% errado**. Nao se repete o erro conhecendo-o.

**Corolario de projeto, novo na campanha e generalizavel:** *quando a incerteza
de um coeficiente e dominada por falta de ajuste, ela se compra com **pontos de
projeto**, nao com **replicas**.* As duas dimensoes do desenho governam coisas
diferentes: **replicas governam a precisao por ponto (o criterio primario);
pontos governam a precisao dos coeficientes da superficie (a interacao).** Por
isso o numero de pontos e fixado no **desenho** (sec.1.2, por detectabilidade) e
a escada opera **so** sobre R.

### 2.3 As demais alternativas consideradas e descartadas

**(c) 1/2 da margem (0,050), o numero do EX-2.** Rejeitada pelo mesmo argumento
do EX-3: a superficie atravessa a margem, e 0,050 deixaria indeterminada uma
faixa larga em torno do cruzamento -- justamente onde vive o entregavel.

**(d) Criterio relativo ao efeito.** Rejeitado, como no EX-3: nao termina (custo
diverge como `1/efeito^2`) e e circular se calculado sobre o efeito observado.
No canto (lambda alto, mix baixo) o efeito previsto e ~+0,01, onde um alvo de
10% relativo exigiria R da ordem de 10^4.

**(e) Alvo sobre o EFEITO TOTAL da interacao no espaco** (`b3 x amplitude_lambda
x amplitude_p_mr`, em unidades de MOS-1, comparavel a margem AC-3). Foi a
alternativa mais atraente porque nao usa numero de piloto nenhum -- o alvo seria
0,025, a mesma constante do primario. **Rejeitada por inviabilidade medida:** a
semi-largura projetada desse total e **0,091 em R=384** e satura em ~0,088;
atingir 0,025 exigiria ~420 pontos de projeto. Registrada aqui porque e a
formulacao *conceitualmente* correta do alvo -- e o que a torna inaplicavel e a
falta de ajuste, nao a sua logica.

## 3. Criterios CONFIRMATORIOS (declarados, NAO gatilhantes)

Nenhum reprova a parada sozinho; sao registrados junto com o veredito.

1. **Interacao detectada:** IC 95% de `b3` (bootstrap dos pontos de projeto)
   **exclui zero**, com **sinal negativo**, e o **teste F de aditividade
   rejeita** (alpha = 0,05). Previsao em sec.8.
2. **Classificacao contra a margem AC-3, por ponto** (herdado do EX-3): cada
   ponto recebe `ci_low > +0,10` (supera), `ci_high < +0,10` (aquem) ou
   indeterminado. **Meta declarada: nenhum indeterminado exceto os adjacentes a
   fronteira prevista.** Com a fronteira agora sendo uma CURVA, "adjacente"
   significa: a menos de 0,05 em p_mr do `p*(lambda)` previsto para o lambda
   daquele ponto.
3. **Fronteira como curva resolvida:** a banda de 95% de `p*(lambda)` (sec.4.4)
   deve ter semi-largura <= 0,05 em p_mr nos lambdas reportados em que a
   fronteira cai **dentro** da faixa realizada.
4. **Q2 do surrogate >= 0,50** (Golbraikh & Tropsha 2002). **Registrar SEMPRE as
   duas leituras**: o `cv.mean.r2` publicado e o **Q2 poolado** calculado por
   fora (sec.7.5b do desenho da campanha). Previsao em sec.8.
5. **Estabilidade entre degraus da escada** (espelha `convergence_stop(0.05,...)`):
   deriva da media <= 0,25 x semi-largura e variacao relativa de media E desvio
   < 5% entre lotes.
6. **Cross-check contra as marginais medidas** (sec.10): a superficie ajustada
   so no EX-4, avaliada ao longo das retas `p_mr = 0,5` e `lambda = 6,0`, deve
   reproduzir os 16 pontos do EX-2 e os 20 do EX-3 -- que sao amostras
   **independentes** (bancos de seed disjuntos). Alvo declarado: **RMSE <= 0,05
   em unidades de MOS-1** contra cada conjunto (= 1/2 da margem AC-3).

## 4. Dimensionamento -- a base empirica

### 4.1 A DISCREPANCIA DOS DOIS MODELOS DE SIGMA, RESOLVIDA ANTES DE DIMENSIONAR

Os dois ajustes medidos discordam onde se cruzam. Verificado nesta sessao:

| Fonte | Previsao em (lambda=6, p_mr=0,5) | erro vs EX-1 |
|---|---|---|
| Ajuste do EX-2 `sigma = 0,3373 * lambda^-0,4613` | 0,1476 | **+17,3%** |
| Ajuste do EX-3 `sigma = 0,0978 * exp(0,5319 p)` | 0,1275 | **+1,4%** |
| **EX-1 MEDIDO (N=200, medida direta)** | **0,1258** | -- |

Os dois modelos discordam entre si em **15,7%**, e o EX-1 da razao ao EX-3.
Tres hipoteses estavam em aberto; foram testadas:

**H1 -- alavancagem dos pontos de lambda baixo no ajuste do EX-2: CONFIRMADA.**
Reajustando sem os pontos de menor lambda:

| pontos removidos | expoente | sigma(6) |
|---|---|---|
| 0 (n=16, publicado) | -0,4613 | 0,1476 |
| 1 (n=15) | -0,4589 | 0,1474 |
| 2 (n=14) | -0,4505 | 0,1465 |
| **3 (n=13)** | **-0,3308** | **0,1345** |
| 4 (n=12) | -0,3405 | 0,1355 |

**H2 -- o que o EX-2 MEDIU perto de lambda=6 (sem modelo): decide a questao.**

| lambda | sigma medido | modelo do EX-2 | razao |
|---|---|---|---|
| 4,018 | **0,2113** | 0,1776 | **1,190** |
| 5,388 | 0,1376 | 0,1551 | 0,887 |
| 6,721 | 0,1435 | 0,1401 | 1,025 |
| 7,950 | 0,1277 | 0,1296 | 0,985 |

Os dois vizinhos de lambda=6 mediram **0,1376 e 0,1435** -- proximos do 0,1258
do EX-1 e do 0,1275 do EX-3, e **longe** do 0,1476 do modelo. O ponto de
lambda=4,018, com sigma 19% acima do proprio ajuste, e o mesmo que amarrou a
parada do EX-2: ele puxa o expoente para baixo e levanta o nivel em lambda=6.

**H3 -- a superficie de sigma nao seria separavel: NAO SUSTENTADA.** Teste
direto no unico conjunto 2-D (ME-1, 32 pontos), acrescentando `log(lambda) x
p_mr` ao ajuste de `log sigma`:

```
F(1,28) = 0.3441, p = 0.5621  ->  NAO rejeita a separabilidade
```

**Conclusao registrada:** a discrepancia **nao** e falta de separabilidade nem
interacao na variancia -- e **artefato do ajuste do EX-2**, cujo expoente e
sustentado por poucos pontos de lambda baixo de alta alavancagem, um deles com
ruido proprio de +19%. **O EX-4, com 32 pontos espalhados em duas dimensoes, e
o primeiro desenho capaz de reajustar a superficie de sigma sem essa
alavancagem, e deve reporta-la no fechamento.**

### 4.2 A superficie de sigma adotada (ancorada em NIVEL, nao em derivada)

Aplica-se a licao do EX-3 -- *um ajuste-piloto transfere melhor em nivel do que
em derivada*: adota-se a **forma separavel** (nao rejeitada por H3), com os
**expoentes medidos com `--config`** (EX-2 para lambda, EX-3 para mix) e o
**nivel ancorado na unica medida direta**, o EX-1:

```
sigma(lambda, p) = 0,12578 * (lambda/6)^-0,4613 * exp(0,5319 * (p - 0,5))
```

| lambda | p=0,00 | p=0,25 | p=0,50 | p=0,75 | p=1,00 |
|---|---|---|---|---|---|
| 2,00 | 0,1600 | 0,1828 | 0,2088 | 0,2385 | **0,2724** |
| 2,25 | 0,1516 | 0,1731 | 0,1977 | 0,2259 | 0,2580 |
| 6,00 | 0,0964 | 0,1101 | **0,1258** | 0,1437 | 0,1641 |
| 18,00 | 0,0581 | 0,0663 | 0,0758 | 0,0865 | 0,0989 |

**Mantem-se o expoente -0,4613 do EX-2 apesar de H1**, e nao o -0,3308 do
reajuste: o expoente mais ingreme preve sigma **maior** em lambda baixo, que e o
lado conservador para dimensionar. A escolha e declarada, nao acidental.

**O pior canto e (lambda BAIXO, mix ALTO)** -- combinacao dos dois piores cantos
do EX-2 e do EX-3, e um canto que nenhum estudo anterior visitou.

### 4.3 PREVISAO A PRIORI DO DIMENSIONAMENTO (a ser confrontada no fechamento)

O canto extremo (2 ; 1) tem sigma = 0,2724, exigindo R = 456. Mas **o LHS nao
sorteia o canto**: com 32 pontos, o ponto de lambda mais baixo cai em [2,0; 2,5]
e nao necessariamente com mix alto. Simulando **5.000 sorteios LHS de 32 pontos**
e avaliando `max` do sigma modelado sobre os pontos:

| percentil do sorteio | sigma do pior ponto | R necessario |
|---|---|---|
| p25 | 0,1971 | 239 |
| **p50** | **0,2132** | **279** |
| p90 | 0,2450 | 369 |
| p95 | 0,2524 | 392 |
| p99 | 0,2625 | 424 |

| degrau | P(satisfaz em TODOS os 32 pontos) |
|---|---|
| R = 192 | **4,5%** |
| **R = 384** | **93,9%** |
| R = 768 | **100,0%** |

> **PREVISAO: R=192 reprova; R=384 ACEITA. O ponto que amarra estara na regiao
> de lambda baixo com mix alto** -- e, pela licao do EX-2 e do EX-3, **preve-se
> errar QUAL ponto** amarra (sigma tem ruido proprio), acertando a **regiao**.

**Margem de erro da propria previsao.** R=768 cobre sigma ate 0,354, isto e,
tolera uma subestimacao de **35%** da superficie de sigma -- alem da dispersao
de +/-25% que o EX-2 mediu. O quarto degrau (1536) e declarado mesmo assim, por
disciplina: **nao se acrescenta degrau depois de ver o resultado**. Preve-se que
nao sera usado.

**Custo estimado.** 32 x (192 + 384) = **18.432 pares** se parar em R=384; 43.008
se for a 768. A ~50 ms/replica (mediana das sessoes EX-2/EX-3) isso da ~15 min de
CPU e poucos minutos de relogio com `--workers 6`. Com 32 futures, o paralelismo
sera o mais efetivo da campanha. **Custo nao e restricao** -- o dimensionamento e
estatistico.

### 4.4 Reconstrucao bilinear consistente das marginais (base das previsoes)

Combinando o que ja foi **medido com `--config`** com a interacao **do piloto**:

- EX-3 (lambda=6): `Delta(6,p) = -0,0060 + 0,4393 p`
- EX-2 (p_mr=0,5): `dDelta/dlambda = -0,00638`
- ME-1 (piloto): `b3 = -0,014776`

Resolvendo, a unica superficie bilinear compativel com as tres e:

```
Delta(L,p) = -0,012048 + 0,001008 L + 0,527956 p - 0,014776 L p
```

| lambda | inclinacao do mix | fronteira p* (Delta = +0,10) |
|---|---|---|
| 2,0 | +0,4984 | **0,2208** |
| 6,0 | +0,4393 (= medido no EX-3) | **0,2413** (= medido no EX-3) |
| 12,0 | +0,3506 | 0,2851 |
| 18,0 | **+0,2620** | **0,3584** |

> **CORRECAO REGISTRADA ANTES DE RODAR.** A previsao de que a fronteira iria a
> **p_mr = 0,696 em lambda=18** (citada no briefing desta sessao) **e
> inconsistente**: ela desloca a reta inteira do EX-3 pela inclinacao
> **marginal** do EX-2 -- que ja contem metade da interacao (`b1 + 0,5 b3`) --
> **e** ainda gira a reta por `b3`, contando a interacao duas vezes. O valor
> consistente com os mesmos tres insumos e **p*(18) = 0,358**. A previsao de
> inclinacao (+0,262 em lambda=18, queda de 40%) **esta correta** e e mantida.
> Este numero nao aparece em nenhum doc versionado -- so no briefing -- entao
> nao ha correcao de doc a fazer, mas a previsao *a priori* do EX-4 e **0,358**,
> e e contra ela que o resultado sera confrontado.

## 5. base_seed nova (20260808), declarada antes de rodar

EX-1 usou 20260804; EX-2, 20260806; EX-3, 20260807. Como
`_crn_seeds(base, design_point_id, n)` deriva o banco de
`SeedSequence(base).spawn(...)`, reusar qualquer uma faria o **ponto de projeto
0 do EX-4 herdar o banco inteiro** do estudo correspondente. O numeral 20260808
e um **rotulo**, nao uma afirmacao de data.

**Ganho colateral, e no EX-4 ele e o mais rico da campanha.** O espaco do EX-4
**contem** o ponto do EX-1 (6 ; 0,5), a reta do EX-2 (p_mr = 0,5) e a reta do
EX-3 (lambda = 6). Com banco disjunto:

- o ponto sorteado mais proximo de (6 ; 0,5) e **replica independente do EX-1**;
- a superficie ajustada, avaliada sobre as duas retas, e **checagem independente
  do EX-2 e do EX-3 inteiros** (36 pontos medidos por outros estudos).

E a validacao cruzada mais forte que a campanha pode produzir, e e o objeto do
confirmatorio 6 da sec.3.

## 6. Aninhamento CRN entre degraus da escada

`_crn_seeds(base, dp_id, 384)[:192] == _crn_seeds(base, dp_id, 192)` para **cada**
ponto de projeto, porque `SeedSequence.spawn(n)` usa chaves 0..n-1 (sec.7.2 do
desenho). Logo a comparacao entre degraus e convergencia sobre a MESMA
sequencia, ponto a ponto -- nao uma segunda amostra independente. **Sera
verificado em dados**, ponto a ponto e seed a seed; se falhar, o protocolo de
parada esta invalido e o `campaign_postprocess.py` sai com codigo 1.

**O CRN NAO pareia entre pontos de projeto** (sec.3.3 do desenho): comparacoes
*entre* niveis de fator nao tem reducao de variancia por pareamento. Isso vale
com forca redobrada aqui, onde a superficie inteira e um objeto entre pontos.

## 7. Esparsidade e degeneracao -- tratamento declarado

**Este e o primeiro estudo da campanha em que o canto lambda -> 2 e
genuinamente alcancavel.** No EX-2 o menor lambda sorteado foi 2,99; no EX-3
lambda estava fixo em 6,0. Com 32 estratos, o mais baixo e **[2,0 ; 2,5]**.

| Resposta | Ausencia significa | Tratamento |
|---|---|---|
| `leak_<motivo>` | zero ocorrencias | **zero-fill** sobre (ponto x replica x config x motivo) |
| `rockets_per_engagement` | replica SEM engajamento | **NAO zero-fill**; media indefinida. Reportar `N_eff` por ponto e por config; parear so onde os dois bracos tem a linha |
| MOS-1 | -- | O agregador emite `0.0` quando `total_arrivals == 0` (`aggregator.py:105`). O par entra no Delta puxando para zero **sem sinalizar**; contado em `sparsity.n_degenerate_zero_delta` |

**A reportar por ponto de projeto, no minimo:** `N_eff` de
`rockets_per_engagement` por config e a contagem de **replicas degeneradas**.

**Previsao a priori -- e desta vez a previsao e de que a esparsidade
MATERIALIZA.** Base: no ME-1 o ponto de lambda = 2,021 teve `N_eff` **23/24 nos
DOIS bracos** (unica vez em toda a campanha em que `N_eff` nao foi cheio), taxa
de 4,17%; os pontos de lambda 2,62 / 3,38 / 3,66 tiveram 24/24.

> **Previsao: `N_eff` < R em 1 a 3 pontos, todos com lambda < 2,6, a uma taxa da
> ordem de 1-4%** -- em R=384 isso e ~4 a 16 replicas sem engajamento no pior
> ponto. **Previsao secundaria: replicas degeneradas > 0 pela primeira vez na
> campanha.** Se `N_eff` sair cheio nos 32 pontos, a salvaguarda continua
> justificada (ela e barata e o precedente do ME-1 e real), mas a previsao terah
> sido **refutada pelo quarto estudo seguido**, e isso deve ser registrado como
> tal.

## 8. Respostas incluidas e PREVISOES A PRIORI POR RESPOSTA

`MOS-1, MOE-1, MOE-3, MOP-1, MOE-2, MOS-2` -- MOS-1 **primeiro** (`primary =
responses[0]` governa o `decision_summary`/`decision_index`). Confirma a lista do
EX-2 e do EX-3, sem ajuste.

- **MOP-2 / MOP-3 excluidas**: constantes de config, variancia exatamente zero.
- **MOS-2**: entra em `--responses` (a tabela mestra a trata pelo ramo
  `config="absolute"`) e recebe bootstrap de **UMA amostra** sobre a media
  **dentro de cada ponto de projeto**. `analysis stats --response MOS-2` levanta
  `ValueError` e **nao sera rodado**.

### 8.1 Previsoes de INTERACAO por resposta (do ajuste bilinear do ME-1)

Cada linha e um teste F de aditividade sobre os 32 pontos do ME-1, re-derivado
nesta sessao. **A previsao e que o EX-4 reproduza o SINAL e a SIGNIFICANCIA de
cada linha, com `--config`:**

| Resposta | `b3` no ME-1 | F(1,28) | p | previsao para o EX-4 |
|---|---|---|---|---|
| **MOS-1** | **-0,01478** | 11,88 | 0,0018 | interacao **negativa** e significativa |
| **MOE-1** | **-0,01145** | 16,81 | 0,00032 | interacao negativa, **a mais forte das cinco** |
| MOE-3 | -0,00784 | 5,69 | 0,0240 | negativa, **a mais fraca** -- pode nao rejeitar |
| **MOE-2** | **+0,00898** | 21,44 | 0,00008 | **POSITIVA** -- sinal oposto ao de MOS-1 |
| **MOP-1** | +4,113 | 1,00 | **0,3261** | **a UNICA aditiva** -- preve-se **nao rejeitar** |

**A previsao mais nitida e a do MOP-1**: e a unica resposta em que os dois
efeitos principais tem sinais opostos entre os estudos marginais (r(lambda) =
**+0,91** no EX-2 contra r(p_mr) = **-0,997** no EX-3) **e** em que o piloto
**nao** detecta interacao. Previsao: os dois efeitos principais se mantem com
esses sinais e **se somam sem termo cruzado** -- Delta MOP-1 maximo no canto
(lambda alto, mix baixo). Se o EX-4 achar interacao significativa em MOP-1, a
leitura de "desempenho de sensor e separavel do cenario" cai.

**MOE-2** e o segundo teste nitido: interacao de sinal **oposto** ao de MOS-1,
coerente com o mecanismo de forma fechada (a queda de MOE-2 e artefato do anel
de amostragem, sec.5.2 do desenho) -- ela nao acompanha o valor de missao.

### 8.2 Previsao sobre a manchete (a interacao em MOS-1)

> **`b3` entre -0,010 e -0,020, IC 95% excluindo zero, teste F rejeitando a
> aditividade.** A inclinacao do mix cai de **+0,4393** (lambda=6, medido no
> EX-3) para **~+0,262** (lambda=18) -- queda de **40%**. A fronteira de
> aplicabilidade sobe de **p* = 0,241** (lambda=6, medido no EX-3) para
> **p* ~ 0,358** (lambda=18), permanecendo **dentro** da faixa amostrada em todo
> o eixo de lambda.

**O que falsifica a interacao do piloto:** se `b3` sair com IC contendo zero, ou
se a inclinacao do mix for estatisticamente a mesma nos dois extremos de lambda,
entao a interacao medida no ME-1 era **artefato de ter rodado sem `--config`**
(fronteira sintetica, sem o bloco `arc` do Spec 030). Esse resultado seria tao
publicavel quanto o outro, e a redacao ja esta comprometida com ele.

### 8.3 Previsao sobre o artefato de Q2 (sec.7.5b) e o gate de evidencia

O artefato de `cv.mean.r2` e **condicional**: morde quando a faixa da resposta e
estreita em relacao ao ruido (EX-2, faixa 1,65x, divergencia 0,71) e nao morde
quando e larga (EX-3, faixa 29,4x, divergencia 0,01). No EX-4 a faixa prevista
vai de ~+0,01 (lambda alto, mix baixo) a ~+0,45 (lambda baixo, mix alto), da
ordem de **30x**. **Previsao: `cv.mean.r2` >= 0,5 e proximo do poolado**, como no
EX-3. O Q2 poolado sera calculado por fora de qualquer forma e as duas leituras
publicadas lado a lado.

**Gate de evidencia:** previsao de que **PASSA** (32 >= 20 pontos com
`n_factors = 2`, Loeppky; pares >> 34, Cohen; Q2 >= 0,5) -- segunda vez na
campanha.

### 8.4 Previsoes de custo e modos de falha

- `rockets_per_engagement` (modernizado): **cresce com p_mr** (confirmado no
  EX-3: 2,478 -> 2,814) e preve-se **cair levemente com lambda** (menos tempo de
  engajamento por alvo sob saturacao). Confianca media no segundo.
- `leak_pending-evaded`: **cai com p_mr** (EX-3: fator 7,3x) e **cresce com
  lambda** (mais chegadas por plataforma). Previsao: **o maior valor do espaco
  no canto (lambda alto, mix baixo)**.
- `leak_winchester` (modernizado): **cai com p_mr** (EX-3 mediu 13x, refutando a
  previsao de que seria plano). Em lambda: no EX-2 era lambda quem o movia.
  Previsao: maximo no canto (lambda **baixo**, mix baixo) -- tail-chase com
  tempo de sobra para esgotar o pente. Confianca baixa, declarada como tal.
- `leak_winchester` (baseline): **zero em todos os pontos** (previsao repetida do
  EX-3, la confirmada).

## 9. Regras de redacao (decididas antes de ver os dados)

**As regras 1 a 5 da sec.9 do `STOPPING_CRITERION.md` do EX-3 sao herdadas
integralmente e sem alteracao**, e sao aqui reproduzidas por referencia:
(1) afirmacao positiva so com IC excluindo zero; (2) IC contendo zero permite
**apenas** afirmacao de limite, nunca "o ganho e zero" (Altman & Bland 1995;
Amrhein, Greenland & McShane 2019); (3) classificacao de cada ponto contra a
margem AC-3 em tres categorias; (4) a manchete de fronteira so se sustenta em
pontos do tipo "aquem"; (5) a localizacao do cruzamento reportada com incerteza
propagada, nunca como valor pontual.

Acrescentam-se tres regras proprias do desenho 2-D:

6. **A fronteira e uma CURVA `p*(lambda)`, com banda.** Sera estimada pela
   superficie bilinear ajustada, `p*(L) = (0,10 - b0 - b1 L) / (b2 + b3 L)`, com
   incerteza propagada pelo **mesmo bootstrap de pontos de projeto** que da o IC
   dos coeficientes (Kleijnen 2015 -- replicas de um ponto sao correlacionadas;
   o ponto e a unidade independente). Reportada como banda, nunca como linha.
7. **Fora da faixa realizada, nao ha fronteira a afirmar.** Onde `p*(lambda)`
   cair fora do intervalo de p_mr efetivamente amostrado, a unica afirmacao
   permitida e *"sem cruzamento da margem na faixa amostrada"*. **Proibido
   extrapolar** a curva para p_mr < min ou > max realizados, ou para lambda fora
   da faixa realizada.
8. **A interacao e reportada NO MODELO EM QUE FOI MEDIDA.** `b3` e dependente de
   modelo -- no ME-1 vale -0,01478 no bilinear e -0,01023 no quadratico. A
   manchete usa o **bilinear**, porque e o modelo do piloto e o unico em que a
   comparacao com -0,014776 e like-for-like; o **quadratico** entra como
   robustez declarada, jamais como manchete substituta. Qualquer afirmacao sobre
   a magnitude da interacao cita o modelo.

## 10. Ferramental -- o que muda e o que NAO muda

**NAO muda:** `scripts/campaign_postprocess.py` e `scripts/make_study_design.py`
ficam **intocados**. O criterio primario (max da semi-largura por ponto) e
calculado pelo `campaign_postprocess.py` exatamente como no EX-2 e no EX-3, sem
uma linha de diferenca -- logo a nao-regressao verificada contra
`offpath_n200.json` (EX-1) e `offpath_r96.json` (EX-2) continua valendo por
construcao.

**O `--factor` aceita UM fator, e com dois variando a decisao e:** rodar o
`campaign_postprocess.py` **duas vezes**, uma por fator. Verificado no codigo
antes de rodar: `--factor` afeta **apenas** (i) o rotulo do valor do fator em
`by_design_point[].<factor>` e (ii) o bloco `trend` (regressao simples contra
aquele fator). O criterio **primario** -- `max_half_width`, `all_points_pass`,
`worst_design_point_id` -- e **identico nas duas execucoes**, porque nao depende
do fator. As duas passagens entregam as duas **marginais medias sobre o espaco
amostrado** (que NAO sao as marginais do EX-2/EX-3: aquelas tem o outro fator
fixo no nominal; estas o tem distribuido uniformemente pelo LHS). Isso sera dito
explicitamente no fechamento.

**MUDA (aditivamente):** um script novo, `scripts/surface_model.py`, **commitado
junto com este arquivo, antes do primeiro lote**. Ele faz o que nenhum
ferramental existente faz: ajusta a superficie `[1, L, p, L*p]` sobre as medias
por ponto de projeto, da IC bootstrap de cada coeficiente (inclusive `b3`), roda
o teste F de aditividade, produz a inclinacao do mix por lambda, a **fronteira
como curva com banda**, o modelo quadratico de robustez e os cross-checks contra
os `offpath_*.json` do EX-2 e do EX-3. Justificativa de nao estender o
`campaign_postprocess.py`: aquele arquivo carrega o criterio de parada de tres
estudos fechados e tem `--assert-baseline` contra dois deles; um arquivo novo e
aditivo por construcao e nao coloca essa garantia em risco. E `scripts/`, sem
spec, como os outros dois.

## 11. O que este estudo NAO conclui

- Nada fora das faixas **REALIZADAS** de lambda e p_mr (a declarar no
  fechamento; os extremos 2 / 18 e 0 / 1 nao serao amostrados).
- Nada sobre **doutrina de emprego** (filosofia de tiro, re-engajamento,
  penalidades dimensionais) -- eixo AN-doutrina, fora da campanha EX-*.
- Nada sobre termos de **ordem superior** alem do quadratico de robustez: com 32
  pontos nao se ajusta uma superficie mais rica sem sobreajustar.
- **A interacao medida e a do modelo bilinear** (sec.9 regra 8).
- O `decision_index` continua ranqueando **AoRs**, nao pontos de projeto nem
  niveis de fator (`report.py:532`) -- a significancia por ponto vem de
  `by_design_point[]` do `offpath_*.json`, sob as regras 1-5.
- MOE-2 continua **sem ruido proprio** (sec.5.2 do desenho): reportada, nunca
  usada como evidencia de precisao superior.
