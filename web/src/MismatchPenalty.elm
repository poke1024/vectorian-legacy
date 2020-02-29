module MismatchPenalty exposing (mismatchPenaltyCurve)

import LineChart
import LineChart.Junk as LineChartJunk
import LineChart.Area as LineChartArea
import LineChart.Axis as LineChartAxis
import LineChart.Junk as LineChartJunk
import LineChart.Dots as LineChartDots
import LineChart.Grid as LineChartGrid
import LineChart.Dots as LineChartDots
import LineChart.Line as LineChartLine
import LineChart.Colors as LineChartColors
import LineChart.Events as LineChartEvents
import LineChart.Legends as LineChartLegends
import LineChart.Container as LineChartContainer
import LineChart.Interpolation as LineChartInterpolation
import LineChart.Axis.Intersection as LineChartIntersection
import LineChart.Axis.Range as LineChartRange
import LineChart.Axis.Title as LineChartTitle
import LineChart.Axis.Ticks as LineChartTicks
import LineChart.Axis.Line as LineChartAxisLine

import Html exposing ( Html, Attribute )
import Msg exposing (..)

type alias MismatchWeightData =
  { words : Float
  , weight : Float
  }

chartContainerConfig : LineChartContainer.Config msg
chartContainerConfig =
  LineChartContainer.custom
    { attributesHtml = []
    , attributesSvg = []
    , size = LineChartContainer.relative
    , margin = LineChartContainer.Margin 0 0 40 60
    , id = "chart-id"
    }

yAxisConfig : LineChartAxis.Config MismatchWeightData msg
yAxisConfig =
  LineChartAxis.custom
    { title = LineChartTitle.default ""
    , variable = Just << .weight
    , pixels = 400
    , range = LineChartRange.window 0 100
    , axisLine = LineChartAxisLine.rangeFrame LineChartColors.gray
    , ticks = LineChartTicks.float 4
    }

chartConfig : Int -> LineChart.Config MismatchWeightData msg
chartConfig size =
  { y = yAxisConfig  -- LineChartAxis.picky 400 "" .weight [0, 0.25, 0.5, 0.75, 1]
  , x = LineChartAxis.picky 700 "" .words (List.map toFloat (List.range 1 size))
  , container = chartContainerConfig  -- LineChartContainer.responsive "chart-id"
  , interpolation = LineChartInterpolation.monotone
  , intersection = LineChartIntersection.default
  , legends = LineChartLegends.none
  , events = LineChartEvents.default
  , junk = LineChartJunk.default
  , grid = LineChartGrid.default
  , area = LineChartArea.stacked 0.5
  , line = LineChartLine.default
  , dots = LineChartDots.default
  }

mismatchPenaltyCurveData : Float -> Float -> MismatchWeightData
mismatchPenaltyCurveData cutoff x
  = MismatchWeightData x (100 * (min 1 (1 - (e ^ -(x / (cutoff / 0.693147))))))

mismatchPenaltyCurve : Int -> Int -> Html Msg
mismatchPenaltyCurve cutoff size
  = let s = (max size cutoff) in
    LineChart.viewCustom (chartConfig s)
    [ LineChart.line LineChartColors.grayLight LineChartDots.none "mismatch-penalty"
      (List.map2 mismatchPenaltyCurveData (List.repeat s (toFloat cutoff)) (List.map toFloat (List.range 1 s)))
    ]
