# Dataset Decision for Hybrid RL-TOPSIS

## Short Answer

Amazon India is not a mistake. It was attractive because it contains several
catalog and review-derived fields that directly support the MCDM side of the
study:

- discounted price
- actual price
- discount percentage
- rating
- rating count
- product category path
- product description
- review title/content aggregates
- reviewer identifiers embedded per product row

Those fields make it natural to derive price, quality, popularity, discount,
risk, review-richness, and category-level criteria. That is why the dataset is
well aligned with an MCDM/reinforcement-learning decision-support study.

The limitation is different: the CSV does not provide reliable temporal
clickstream/purchase sequences. Therefore it is weaker as a pure recommender
systems benchmark.

## Recommended Two-Dataset Strategy

Use two complementary datasets:

### Dataset A: Amazon India Product/Review Catalog

Role in the paper:

- primary MCDM-rich product catalog experiment;
- demonstrates the proposed method on real product metadata;
- supports price/quality/popularity/discount/rating/category criteria;
- suitable for explaining criterion-level and behavioral-fusion mechanisms.

Main limitation:

- no temporal clickstream or purchase labels;
- behavioral layer must be described as an explicit profile-level feedback
  model, not as observed purchases.

### Dataset B: McAuley/UCSD Amazon Home & Kitchen 5-Core

Role in the paper:

- external recommender-style validation;
- real user-item interactions;
- real timestamps;
- recognized dataset family in recommender-system literature.

Main limitation:

- fewer explicit commerce fields than Amazon India;
- MCDM criteria must be derived from metadata and aggregate review behavior.

## Datasets Considered

- Amazon India sales/reviews CSV: best match for MCDM criteria, weaker for
  temporal recommendation.
- McAuley/UCSD Amazon reviews: stronger recommender benchmark and broadly
  accepted, weaker direct commerce-feature richness.
- MovieLens 1M: very widely accepted benchmark, but domain mismatch is large
  because it has movies rather than products, prices, discounts, or commerce
  criteria.
- Yelp Open Dataset: strong real reviews and business metadata, but download is
  large and domain is restaurants/businesses rather than product catalogs.
- RetailRocket: strong implicit e-commerce events, but item metadata for MCDM
  criteria is comparatively limited.

## Current Decision

For the Hybrid RL-TOPSIS package, keep Amazon India as the main MCDM-rich dataset and add McAuley Amazon
Home & Kitchen as the external real-interaction validation branch.

If both agree directionally, the paper can claim:

> The method is evaluated on a criterion-rich Amazon India product catalog and
> externally checked on a widely used real Amazon interaction dataset.

The current results are intentionally mixed in this sense: Amazon India supports
the criterion-rich Hybrid RL-TOPSIS mechanism, while McAuley/Home and the
collaborative/deep checks delimit the interaction-rich regimes where CF,
popularity, graph, or sequential methods should be preferred. This is reported
as a scope boundary rather than hidden.



