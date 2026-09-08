# Merchmix: reviewable garment concepts

Design preview: approval required

Existing-style forecasts are for 2020-09-21 to 2020-10-18. New concept sales have not been forecast.

## Forecasting evidence

Held-out test MAE: CatBoost 15.2722; naive 24.6964. MAE reduction: 38.2%. RMSE: 62.8647 versus 80.8710. The exported model is a subsequent final refit.

catboost has the lowest MAE on the common validation population and was selected using validation MAE. Test metrics are a separate held-out evaluation, not a selection criterion. naive has the lowest MAE on the separate ETS sample. ETS sample errors cannot be compared directly with the full-population tree errors. No XGBoost test result is supplied. The exported model is the subsequent final refit.

- Article 0673677002: Forecast sales are 1,667.18 units for the next four weeks. Recent four-week sales were 1,632.00 units. The forecast is 2.16% above recent sales. This existing style-colour is winner rank 1 under the configured sales, growth and recent-customer score. The score is a business heuristic, not a probability. The new garment concept has no sales forecast.
- Article 0865799006: Forecast sales are 1,664.99 units for the next four weeks. Recent four-week sales were 1,323.00 units. The forecast is 25.85% above recent sales. This existing style-colour is winner rank 2 under the configured sales, growth and recent-customer score. The score is a business heuristic, not a probability. The new garment concept has no sales forecast.
- Article 0915529001: Forecast sales are 2,016.44 units for the next four weeks. Recent four-week sales were 2,227.00 units. The forecast is 9.45% below recent sales. This existing style-colour is winner rank 3 under the configured sales, growth and recent-customer score. The score is a business heuristic, not a probability. The new garment concept has no sales forecast.

## Source evidence review

Catalogue-grounded corrections and assistant photo assessment; these are not a new agent response. Other source descriptors remain model estimates and need design-preview review. The original agent response is retained in source_observations_raw.json.

- Article 0673677002, sleeves: full_sleeves → straight_sleeves. Photo assessment: sleeves appear straight, without deliberate puff or balloon volume.
- Article 0673677002, surface: ribbed → plain. Photo assessment: the main body appears plain; ribbed trims are a separate detail.
- Article 0865799006, leg: straight_leg → tapered_leg. Catalogue description: wide, tapered legs.
- Article 0865799006, waist: mid_waist → high_waist. Catalogue description: high waist.
- Article 0865799006, length: regular → ankle_length. Catalogue description: ankle-length trousers.



Winner score: 40% forecast-sales percentile, 30% adjusted historical-growth percentile, 30% recent-customer percentile. These are business choices, not validated success probabilities.

## 1. Blue sweater concept 1

Source article 0673677002 | catalogue name: Henry polo. (1) | selected source colour: Black

![Original selected product](references/0673677002.jpg)

Original catalogue description (quoted; applies to the existing product):

> Jumper in a soft, fine knit with a ribbed polo neck and ribbing at the cuffs and hem.

### Proposed design

Proposed colour: blue.
- Neckline: high neck → crew neck.
- Surface: plain → ribbed.

Retained visual source features: sleeves: straight sleeves; length: regular; hem: straight hem.

Design hypothesis: explore these two visible variations while retaining the listed source features. Forecasting selected the source style; it does not establish that these variations will increase sales.

## 2. Black trousers concept 2

Source article 0865799006 | catalogue name: Pink HW barrel | selected source colour: Light Beige

![Original selected product](references/0865799006.jpg)

Original catalogue description (quoted; applies to the existing product):

> 5-pocket, ankle-length trousers in washed cotton twill with a high waist and wide, tapered legs with decorative seams at the hems.

### Proposed design

Proposed colour: black.
- Leg: tapered leg → wide leg.
- Detail: no added detail → patch pockets.

Retained visual source features: waist: high waist; length: ankle length.

Design hypothesis: explore these two visible variations while retaining the listed source features. Forecasting selected the source style; it does not establish that these variations will increase sales.

## 3. Pink sweater concept 3

Source article 0915529001 | catalogue name: Liliana | selected source colour: Beige

![Original selected product](references/0915529001.jpg)

Original catalogue description (quoted; applies to the existing product):

> Jumper in a soft, fine knit containing some wool with a square neckline and long puff sleeves. Wide ribbing at the cuffs and hem. The polyester content of the jumper is recycled.

### Proposed design

Proposed colour: pink.
- Neckline: square neck → boat neck.
- Sleeves: full sleeves → short sleeves.

Retained visual source features: length: regular; hem: straight hem; surface: plain.

Design hypothesis: explore these two visible variations while retaining the listed source features. Forecasting selected the source style; it does not establish that these variations will increase sales.

## Workflow

A manager agent delegates MCP evidence retrieval and numerical assessment, source-photo observation, and design proposal to specialist tools. The design specialist receives the complete reusable skill plus the constrained visual vocabulary. A user approves the saved source observations, colours, changes and compiled image prompt. A second manager phase delegates one reference-image edit and independent visual observation. The visual reviewer receives the generated pixels without the proposed design values; Python compares its observations with the approved plan. A human reviews the final image. Raw agent responses, MCP calls, approvals, prompts, receipts and file hashes are saved. No automatic image regeneration occurs.

## Limitations

- These are observed sales forecasts for existing style-colour groups, not new designs.
- Inventory availability and unconstrained demand cannot be separated with these data.
- Recent customer counts are observations, not customer forecasts.
- The winner-score weights are a business heuristic and have not been prospectively validated.
- Forecasting evidence does not establish causal effects of garment design features.
- Numerical forecast explanations are checked; broader creative claims still need review.
- Source-photo and generated-image observations are model estimates and may be wrong; unclear attributes remain explicit.
- Catalogue fibre and material statements describe the original product only. The new concepts have no verified composition, fit, comfort, function or environmental benefit.
- The visual vocabulary is deliberately limited; fine construction details and manufacturing feasibility need specialist review.
- Only two test forecast dates and a separate 600-row ETS sample were evaluated.
- Human approval records a design decision, not proof of future demand or manufacturing feasibility.
- Raw H&M training data and the predictive modelling notebook are separate inputs to reproduce model training.
