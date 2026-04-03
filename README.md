# Klike Data Science Challenge — Solução

**Candidata:** Maria Moura
**Entrega:** 03/04/2026

---

## Visão Geral

Este repositório contém a solução completa para o Desafio de Ciência de Dados da **Millai / Klike**. O objetivo é analisar 500 campanhas de anúncios em vídeo, construir um modelo preditivo para o `klike_score` e desenvolver um motor de recomendações acionável para equipes de marketing.

---

## Estrutura do Repositório

```
.
├── solution.ipynb                  # Notebook principal com toda a solução
├── klike_challenge_dataset.csv     # Dataset original (500 campanhas)
├── recommendations_output.csv      # Recomendações geradas para todas as campanhas
├── predictions_output.csv          # Previsões do klike_score para todas as campanhas
├── challenge-doc.md.docx           # Documento original do desafio
└── README.md                       # Este arquivo
```

---

## Como Rodar o Projeto

### Pré-requisitos

- Python 3.9+
- Jupyter Notebook ou Google Colab

### Instalação local

```bash
# Clone o repositório
git clone <url-do-repositório>
cd milei_take_home_challenge

# Instale as dependências
pip install pandas numpy matplotlib seaborn scikit-learn shap

# Abra o notebook
jupyter notebook solution.ipynb
```

### Google Colab

1. Faça upload de `solution.ipynb` e `klike_challenge_dataset.csv` no Colab
2. No início do notebook, descomente o bloco de upload:
   ```python
   from google.colab import files
   uploaded = files.upload()
   ```
3. Execute todas as células de cima para baixo (`Runtime > Run all`)

---

## Estrutura do Notebook

| Seção | Conteúdo |
|-------|----------|
| 1. Visão Geral do Projeto | Resumo do desafio, variável-alvo, métricas e restrições |
| 2. Configuração e Importações | Instalação de dependências e importações |
| 3. Carregamento dos Dados | Leitura do CSV, shape, dtypes, estatísticas descritivas |
| 4. EDA | Valores ausentes, distribuições, outliers, correlações, padrões por plataforma, duração ideal |
| 5. Pré-processamento | Imputação, engenharia de features, encoding, divisão treino/teste |
| 6. Modelagem | Ridge, Random Forest, Gradient Boosting com GridSearchCV e validação cruzada |
| 7. Avaliação | Comparação de modelos, SHAP, gráficos de resíduos |
| 8. Motor de Recomendações | Engine completo + demo em 3 campanhas + exportação CSV |
| 9. Conclusões | Descobertas, premissas, visão de produto (Partes 1–4) |

---

## Resultados Principais

### Modelagem

| Modelo | RMSE | MAE | R² |
|--------|------|-----|----|
| Regressão Ridge (baseline) | ~10.5 | ~8.4 | ~0.54 |
| Random Forest | ~7.8 | ~6.1 | ~0.76 |
| **Gradient Boosting** | **~6.5** | **~5.2** | **~0.82** |

> O **Gradient Boosting** foi o modelo vencedor. Validação cruzada 10-fold confirmou robustez com RMSE ≈ 6.5 ± 0.8.

### Top features preditoras (SHAP)

1. `engagement_rate` — maior driver isolado do klike_score
2. `roas` — retorno sobre investimento sinaliza qualidade do criativo
3. `avg_watch_time_s` / `watch_ratio` — quanto do vídeo é assistido
4. `has_hook` — presença de gancho nos primeiros segundos
5. `ctr` — taxa de cliques como proxy de relevância

### Insights de EDA para Marketing

| Descoberta | Implicação |
|-----------|-----------|
| TikTok tem os maiores scores e engagement | Criativos TikTok estão mais alinhados ao modelo Klike |
| Hooks aumentam o klike_score em ~18 pts no Meta | Priorizar gancho nos primeiros 3s em campanhas Meta |
| Duração ideal: 11–20s no TikTok, 21–45s no LinkedIn | Estratégia de duração deve ser específica por plataforma |
| Alta densidade de texto penaliza em todas as plataformas | Menos texto = melhor pontuação |
| Formato vertical domina TikTok/Meta; horizontal no LinkedIn | Adaptar formato ao padrão da plataforma |
| Rosto humano eleva engagement em ~35% para faixa 25–34 | Considerar criativos estilo UGC |

---

## Motor de Recomendações

O engine recebe os dados de uma campanha (uma linha do dataset) e retorna recomendações ordenadas por impacto estimado no `klike_score`.

### Exemplo de uso

```python
campaign = df_raw[df_raw['campaign_id'] == 'KLK-0004'].iloc[0].to_dict()
recs = recommend(campaign, top_n=3)

# Saída:
# #1 (+9.1) Switch format from quadrado to vertical on TikTok. +9.1 score.
# #2 (+7.9) Add subtitles / captions. On TikTok, +7.9 klike_score points; CTR ~11%.
# #3 (+7.7) Increase video duration from 27s to ~38s on TikTok.
```

### Características do engine

- **Contextual**: considera plataforma, formato e atributos atuais da campanha
- **Quantificado**: cada sugestão indica o ganho estimado em pontos de klike_score e variação de CTR quando relevante
- **Ancorado em dados**: os uplift são calculados a partir dos padrões observados no dataset, não em heurísticas
- **Ordenado por impacto**: recomendações com maior retorno potencial aparecem primeiro
- **Cobre 4 dimensões**: atributos booleanos (hook, face, CTA, legenda), duração, densidade de texto e formato

---

## Parte 4 — Visão de Produto

### Q1: Features adicionais a partir do vídeo original

Se tivesse acesso ao arquivo de vídeo, extrairia:

| Categoria | Features | Justificativa |
|-----------|----------|---------------|
| **Visual** | Emoção facial (DeepFace), destaque de logo (YOLO), taxa de cortes de cena, paleta de cores | Engajamento emocional e lembrança de marca |
| **Áudio** | Velocidade da fala, sentimento, energia musical (librosa), proporção de silêncio | Qualidade do hook e ritmo do criativo |
| **Movimento** | Variância do fluxo óptico, intensidade de movimento nos primeiros 3s | Thumb-stop rate, retenção |
| **Texto sobreposto** | Timing das legendas, tamanho relativo da fonte, posição e timing do CTA | Compreensão, acessibilidade |
| **Multimodal** | Embeddings CLIP de keyframes, score de alinhamento vídeo-linguagem | Qualidade semântica do conteúdo |

### Q2: Arquitetura de produção

```
[App cliente] → POST /recommend {dados_campanha}
      → Serviço FastAPI (stateless)
           → Lookup tables em cache Redis
           → Inferência do modelo (ONNX para baixa latência)
           → JSON com recomendações ordenadas
      → Postgres (log de requisições para auditoria)
      → Feature store assíncrona (Feast/Tecton) → pipeline de retreinamento semanal
```

**Decisões arquiteturais:**
- **FastAPI + Redis**: inferência < 50ms P95 com escalonamento horizontal via Kubernetes
- **ONNX**: serialização do modelo para velocidade máxima sem dependência do sklearn em produção
- **Retreinamento semanal**: Airflow/Prefect recomputa tabelas de referência conforme novos dados chegam
- **A/B testing**: recomendações em modo shadow; mede uplift real no klike_score quando adotadas
- **Dois modelos**: pré-lançamento (só atributos criativos) e pós-lançamento (inclui KPIs)

### Q3: O que faria com mais tempo

1. **Recomendações baseadas em SHAP por campanha** — mais principiado do que lookup tables; cada recomendação seria diretamente derivada do impacto marginal estimado pelo modelo
2. **Modelo pré-lançamento** — regressor usando apenas atributos criativos disponíveis antes da veiculação
3. **Pipeline de análise de vídeo** — extrair as features multimodais descritas no Q1 com MTCNN, librosa e CLIP
4. **Inferência causal** — usar variáveis instrumentais ou DiD para separar correlação de causalidade nos efeitos de cada atributo
5. **Dashboard interativo** — protótipo Streamlit/Dash onde o time de marketing insere uma especificação criativa e recebe pontuação + recomendações em tempo real
6. **Modelos segmentados por plataforma** — Meta, TikTok e LinkedIn têm dinâmicas muito distintas; modelos separados provavelmente capturariam melhor as interações específicas de cada plataforma

---

## Premissas Adotadas

- **KPIs disponíveis na inferência**: colunas como CTR, ROAS, conversões são tratadas como disponíveis (campanha já veiculada). Em produção, um modelo pré-lançamento seria necessário para uso antes da veiculação.
- **`has_subtitle` NaN = ausente**, não False — imputado pela moda por plataforma.
- **Outliers mantidos** em impressions/spend/revenue — transformações logarítmicas reduzem sua influência sem perda de informação.
- **`roas=0`** reflete campanhas de awareness sem objetivo de conversão direta — mantidos como estão.
- `random_state=42` em todos os pontos com aleatoriedade para garantir reprodutibilidade completa.

---

## Dependências

```
pandas>=1.5
numpy>=1.23
matplotlib>=3.6
seaborn>=0.12
scikit-learn>=1.2
shap>=0.41
```
