# Análise e Previsão de Churn de Clientes de Telecomunicações

## 📖 Visão Geral

Este projeto foca na análise de dados de uma empresa de telecomunicações para identificar os principais fatores que levam à evasão de clientes (*churn*). 
O objetivo final é construir e avaliar um modelo de Machine Learning capaz de prever quais clientes têm maior probabilidade de cancelar seus serviços, permitindo que a empresa tome ações proativas para retê-los.

A retenção de clientes é crucial para a sustentabilidade de um negócio, pois adquirir novos clientes geralmente custa mais do que manter os existentes.

## 📊 Dataset

O conjunto de dados utilizado neste projeto é o **"Telco Customer Churn"**, disponível publicamente no Kaggle.

* **Fonte:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
* **Descrição:** O dataset contém informações sobre 7043 clientes, com 21 colunas (features) que descrevem:
    * **Informações Demográficas:** Gênero, faixa etária, se possui parceiro(a) e dependentes.
    * **Serviços Assinados:** Tipo de serviço de telefonia, internet, online security, online backup, etc.
    * **Informações da Conta:** Tempo de contrato (*tenure*), tipo de contrato (mensal, anual), forma de pagamento e valores cobrados (mensal e total).
    * **Variável Alvo:** A coluna `Churn`, que indica se o cliente cancelou ou não o serviço.

## 🚀 Metodologia

O projeto foi desenvolvido seguindo as etapas padrão de um ciclo de vida de ciência de dados:

1.  **Análise Exploratória de Dados (EDA):**
    * Investigação da distribuição das variáveis.
    * Análise da correlação entre as features e a variável alvo (`Churn`).
    * Visualização de dados para identificar padrões e insights, como a relação entre o tipo de contrato e a taxa de churn.

2.  **Pré-processamento e Engenharia de Features:**
    * Tratamento de valores ausentes (se houver).
    * Transformação de variáveis categóricas em numéricas (ex: *One-Hot Encoding* para colunas como `InternetService` e *Label Encoding* para colunas binárias).
    * Normalização/Padronização de features numéricas (`tenure`, `MonthlyCharges`) para otimizar o desempenho de certos algoritmos.

3.  **Treinamento e Validação dos Modelos:**
    * Divisão do dataset em conjuntos de treino e teste (`train_test_split`).
    * Treinamento de diferentes algoritmos de classificação, como:
        * Regressão Logística (como baseline)
        * Árvore de Decisão
        * Random Forest
        * XGBoost / LightGBM
    * Utilização de validação cruzada para garantir a robustez dos resultados.

4.  **Avaliação de Modelos:**
    * Como o dataset pode ser desbalanceado (mais clientes que não cancelam), a **Acurácia** não é a única métrica relevante. Foram utilizadas:
        * **Matriz de Confusão**
        * **Precisão (Precision)**
        * **Recall (Sensibilidade)**
        * **F1-Score**
        * **Curva ROC e AUC**

## 🎯 Resultados

O modelo que apresentou o melhor desempenho geral foi o **[NOME DO SEU MELHOR MODELO, ex: XGBoost]**, alcançando os seguintes resultados no conjunto de teste:

* **Acurácia:** `[VALOR, ex: 0.85]`
* **F1-Score:** `[VALOR, ex: 0.65]`
* **AUC:** `[VALOR, ex: 0.88]`

As features mais importantes para prever o churn, de acordo com o modelo, foram:
1.  `Contract` (Tipo de Contrato)
2.  `tenure` (Tempo como Cliente)
3.  `InternetService` (Tipo de Serviço de Internet)
4.  `MonthlyCharges` (Cobrança Mensal)

**Conclusão:** Clientes com contratos mensais, pouco tempo de casa e serviços de internet de fibra óptica tendem a ter uma probabilidade de churn significativamente maior.
## 💻 Tecnologias Utilizadas

* **Linguagem:** Python 3
* **Bibliotecas:**
    * Pandas (para manipulação de dados)
    * NumPy (para computação numérica)
    * Matplotlib e Seaborn (para visualização de dados)
    * Scikit-learn (para pré-processamento, modelagem e avaliação)
    * XGBoost / LightGBM (para modelos avançados)
    * Jupyter (para desenvolvimento interativo)
 
## 👤 Autor

* **[Leonardo Kirian]**
* **LinkedIn:** [Linkedin](https://www.linkedin.com/in/leonardo-kirian-626017131/)
* **GitHub:** [GitHub](https://github.com/LeoKirian/)
