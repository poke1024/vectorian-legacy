module Msg exposing (..)

import Server exposing (CostCombineFunction, SimilarityMeasure)

type QuerySettingsMsg =
  IgnoreDeterminers Bool |
  EnableElmo Bool |
  AnnotatePOS Bool |
  AnnotateDebug Bool |
  PosWeighting (Maybe Float) |
  PosMismatch (Maybe Float) |
  MismatchLengthPenalty (Maybe Int) |
  SubmatchWeight (Maybe Float) |
  IDFWeight (Maybe Float) |
  Bidirectional Bool |
  SimilarityThreshold (Maybe Float) |
  SimilarityFalloff (Maybe Float) |
  UpdateSimilarityMeasure (Maybe SimilarityMeasure) |
  CostCombine CostCombineFunction |
  MixEmbedding (Maybe Float)

type Msg =
  NoOp |

  StartSearch |
  AbortSearch |

  QueryKeyDown Int |
  QueryInput String |
  UpdateQuerySettings QuerySettingsMsg |

  ReceiveDataFromServer String

