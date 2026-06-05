# Detecção de Fraudes em Transações de Cartão de Crédito

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-EB5E28?style=flat)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-interpretabilidade-0096C7?style=flat)
![CRISP-DM](https://img.shields.io/badge/Metodologia-CRISP--DM-023047?style=flat)
![Licença](https://img.shields.io/badge/Licença-ODbL-lightgrey?style=flat)

> Modelo de machine learning que identifica fraudes em transações de cartão de crédito gerando **USD 849 mil de saldo líquido positivo**, com **AUC-ROC de 0.99** e apenas **212 falsos positivos** em mais de 360 mil transações de teste.

---

## Resultado em uma frase

Treinado sobre **1,8 milhão de transações**, o modelo intercepta a grande maioria das fraudes sem bloquear clientes legítimos em excesso. Traduzido para o que importa ao negócio: **USD 977 mil em fraudes evitadas** menos **USD 128 mil em bloqueios indevidos**, resultando em **USD 849 mil de economia líquida**.

| Métrica | Resultado |
|---------|-----------|
| Saldo financeiro líquido | **USD 849.100** |
| Fraudes evitadas | USD 977.056 |
| Legítimas bloqueadas | USD 127.955 |
| AUC-ROC | 0.99 |
| F1-Score (classe fraude) | 0.81 |
| Falsos positivos | 212 |
| Threshold ótimo | 0.86 |

---

## Sobre o projeto

O mercado de pagamentos digitais movimenta trilhões de reais por ano, e identificar fraudes em tempo real sem prejudicar a experiência do cliente é um desafio central para qualquer empresa do setor. Regras manuais não escalam: fraudadores se adaptam rápido e qualquer regra fixa fica obsoleta.

Este projeto constrói um detector de fraudes baseado em aprendizado de máquina, seguindo a metodologia **CRISP-DM** de ponta a ponta. O diferencial não está apenas na performance estatística, mas na tradução de cada decisão técnica em impacto financeiro mensurável, com fundamentação estatística rigorosa em cada etapa.

**Dataset:** [Credit Card Transactions Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/kartik2112/fraud-detection), licença ODbL.

---

## Metodologia CRISP-DM

O projeto está organizado nas seis fases do CRISP-DM:

### 1. Entendimento do Negócio

O objetivo é maximizar o saldo financeiro: priorizar Recall alto (não deixar fraudes passarem) sem comprometer a Precisão (evitar bloquear clientes legítimos). A acurácia é descartada como métrica logo de início, pois com 99,42% de transações legítimas, um modelo que classificasse tudo como legítimo teria 99,42% de acurácia e seria inútil.

### 2. Entendimento dos Dados

A exploração combina análise visual com diagnóstico estatístico formal. As fraudes se concentram em transações de alto valor e na madrugada, com categorias online liderando as taxas de risco.

**Desbalanceamento das classes** — apenas 0,58% das transações são fraudes, o que define toda a estratégia de modelagem.

![Desbalanceamento das classes](01_desbalanceamento.png)

**Valor das transações** — fraudes têm valor médio 8 vezes maior que transações legítimas.

![Valor das transações por classe](02_valor_transacoes.png)

**Taxa de fraude por categoria** — categorias online como shopping_net e misc_net lideram.

![Fraude por categoria](03_fraude_por_categoria.png)

**Perfil das fraudes** — cruzamento de gênero e faixa de valor.

![Perfil das fraudes](04_perfil_fraudes.png)

#### Diagnóstico estatístico

Mais do que gráficos, esta fase aplica testes estatísticos formais para fundamentar cada decisão posterior.

**Correlação de Pearson** entre as variáveis numéricas e a fraude.

![Correlação de Pearson](05a_correlacao.png)

**Correlação de Spearman** — robusta a assimetria, mais adequada a dados financeiros, com threshold formal de 0.85 para detecção de redundância.

![Correlação de Spearman](05e_spearman.png)

**Diagnóstico de multicolinearidade (VIF)** — confirma quantitativamente a redundância severa entre as coordenadas geográficas (VIF acima de 24.000 para longitude), justificando sua remoção.

![VIF](05f_vif.png)

**Diagnóstico de normalidade (Q-Q plot)** — comprova que `amt` e `city_pop` são fortemente assimétricas, validando o uso de Spearman e a escolha do XGBoost.

![Normalidade](05g_normalidade.png)

**Significância estatística (OLS diagnóstico)** — teste de hipótese por coeficiente. Apenas `amt` apresenta significância linear absoluta; as coordenadas confirmam o resultado do VIF.

![P-valores OLS](05h_pvalores.png)

### 3. Preparação dos Dados

As decisões de preparação seguem diretamente os diagnósticos da fase anterior:

- **Remoção das coordenadas brutas** (`lat`, `long`, `merch_lat`, `merch_long`), fundamentada pelo VIF e pelo OLS.
- **Feature engineering:** criação de `distancia` (entre titular e estabelecimento), `hora`, `dia_semana` e `idade`.
- **Encoding sem data leakage:** o `LabelEncoder` é treinado apenas no conjunto de treino e aplicado ao teste, prevenindo vazamento de informação do futuro.

### 4. Modelagem

Comparação de estratégias de balanceamento (SMOTE vs scale_pos_weight) e de algoritmos, com validação rigorosa.

**Comparação dos modelos e curva ROC.**

![Comparação dos modelos](06_comparacao_modelos.png)

**Matrizes de confusão** dos três modelos avaliados.

![Matrizes de confusão](07_matrizes_confusao.png)

**Importância das features** no modelo XGBoost.

![Feature importance](08_feature_importance.png)

**Cross-validation estratificado** — confirma a estabilidade do desempenho entre diferentes divisões dos dados.

![Cross-validation](11_cross_validation.png)

**Curva de aprendizado** — validação converge em F1 de 0.857 com gap controlado de 0.14, indicando modelo bem calibrado com leve overfitting típico do XGBoost.

![Curva de aprendizado](12_learning_curve.png)

#### Interpretabilidade com SHAP

Indo além da importância nativa, o SHAP explica a direção e a magnitude do efeito de cada variável, inclusive por predição individual.

**Summary plot** — `amt` domina; valores altos empurram fortemente para fraude.

![SHAP summary](13a_shap_summary.png)

**Beeswarm** — distribuição completa dos valores SHAP por feature.

![SHAP beeswarm](13b_shap_beeswarm.png)

**Waterfall** — explicação passo a passo de uma fraude individual, útil para auditoria e para analistas de risco.

![SHAP waterfall](13c_shap_waterfall.png)

#### Experimentos de refinamento

Três hipóteses nascidas dos diagnósticos estatísticos foram testadas: remoção de `city_pop`, log-transform em `amt` e log-transform em `city_pop`. Nenhuma melhorou o modelo, e esse é um resultado válido: remover `city_pop` piorou o saldo (o SHAP já mostrava seu valor não linear), e as transformações logarítmicas não tiveram efeito porque o XGBoost é invariante a transformações monótonas.

![Experimentos](14_experimentos.png)

### 5. Avaliação

O modelo campeão (XGBoost com scale_pos_weight, otimizado) é avaliado contra o objetivo de negócio definido na fase 1.

**Otimização do threshold** — em vez do padrão 0.5, o threshold é ajustado para maximizar o saldo financeiro.

![Análise de threshold](09_threshold_analysis.png)

**Impacto financeiro por estratégia** — comparação do saldo líquido entre as abordagens.

![Saldo por cenário](10_saldo_por_cenario.png)

Todos os critérios de sucesso foram atendidos: Recall alto, Precisão preservada, AUC-ROC de 0.99 e saldo líquido positivo de USD 849 mil.

### 6. Implantação

Documentação das considerações para colocar o modelo em produção: requisitos de latência para detecção em tempo real, encapsulamento do pipeline de pré-processamento, threshold configurável pelo negócio, monitoramento de drift e retreino periódico com feedback dos analistas.

---

## Stack técnica

| Categoria | Ferramentas |
|-----------|-------------|
| Linguagem | Python 3.10 |
| Manipulação de dados | Pandas, NumPy |
| Machine learning | XGBoost, scikit-learn |
| Balanceamento | imbalanced-learn (SMOTE, RandomUnderSampler) |
| Estatística | statsmodels, SciPy |
| Interpretabilidade | SHAP |
| Visualização | Matplotlib, Seaborn |

---

## Como executar

```bash
# clone o repositório
git clone https://github.com/oporaxuao/fraud-detection-xgboost.git
cd fraud-detection-xgboost

# instale as dependências
pip install pandas numpy scikit-learn xgboost imbalanced-learn statsmodels scipy shap matplotlib seaborn

# abra o notebook
jupyter notebook Semantix_CRISP-DM.ipynb
```

Os dados podem ser baixados na [página do dataset no Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection). Recomenda-se executar com `Kernel > Restart & Run All` para reproduzir todos os resultados na ordem correta.

---

## Autor

**João Alfredo de Sousa Siqueira**
Cientista de Dados

[![LinkedIn](https://img.shields.io/badge/LinkedIn-oporaxuao-0077B5?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/oporaxuao)
[![GitHub](https://img.shields.io/badge/GitHub-oporaxuao-181717?style=flat&logo=github&logoColor=white)](https://github.com/oporaxuao)
