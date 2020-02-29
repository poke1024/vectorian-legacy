#!/bin/bash
cd "$(dirname "$0")"
elm make src/Vectorian.elm --output=../srv/tornado/static/elm.js
