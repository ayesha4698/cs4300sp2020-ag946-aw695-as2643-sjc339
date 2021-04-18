
"""

INFO 4300: Language and Information
Final Project: Color Palette Generator

Information Retrieval (IR) Main Scoring + Ranking Function

"""

import csv
from app.irsystem.controllers.IR_helpers import *
from app.irsystem.controllers.search_controller import *



# dataset globals
cymColorsInvInd = {}
cymData = {}
# /Users/ayesha/cs4300sp2020-ag946-aw695-as2643-sjc339/app/irsystem/controllers/IR_main.py
with open('data/Cymbolism.csv', mode='r') as infile:
    reader = csv.reader(infile)
    cymData = {rows[0]:rows[1:] for rows in reader}

for i in range(len(cymData['word'])):
    color = cymData['word'][i]
    colorName = color[color.index(' ')+2:]
    cymColorsInvInd[colorName] = i


def getPalettes(keywords, reqColors, energy, numColors):
    """
    Returns a list of palettes sorted from highest to lowest ranked. Calls the
        following backend/IR helper functions:

        1. input_to_color
        2. scorePalettes

    Params: keywords    Cymbolism words matched to user input [List of Strings]
            reqColors   list of user-inputted clean hexcode color [List of String]
            energy      user-input on the muted to bright scale [Int]
            numColors   number of colors the user wants in their palette [Int]
    """

    ranked = []
    
    palettes = input_to_color(keywords, reqColors, energy, numColors)
    scored = scorePalettes(palettes, keywords, reqColors)

    sortedScored = sorted(scored.items(), key=lambda scored: scored[1][1])

    for tup in sortedScored:
        ranked.append(tup[1][0])

    return ranked


def scorePalettes(palettes, keywords, reqColors):
    """
    Returns a new dictionary that scores and ranks each palette based on the
        following factors and weights:

        1. RGB Euclidian Distance to Required Color                        25%
        2. HSV Cartesian Distance to Required Color                        25%
        3. Delta-E Distance to Required Color?                             0%
        4. Cymbolism Keyword Close Color Percentages                       50%

    Final dictionary will be of the following format:
        { paletteID: (List of Colors in Palette, Score),
          ...
          paletteID: (List of Colors in Palette, Score) }

    Params: palettes    palette IDs to Lists of clean hexcodes [Dict of Lists of Strings]
            keywords    Cymbolism words matched to user input [List of Strings]
            reqColors   list of user-inputted clean hexcode color [List of String]
    """

    # preallocate
    scoreDict = {}
    rgbDists = {}
    hsvDists = {}
    percDists = {}

    # weights
    rgbW = .25
    hsvW = .25
    percW = 0
    keyW = .5

    # maximums
    maxRGB = colorDiff((0,0,0), (255,255,255), 'rgb')
    maxHSV = colorDiff((0,0,0), (0,0,100), 'hsv')
    maxPerc = colorDiff(convertColor((0,0,0), 'rgb', 'lab'),
        convertColor((255,255,255), 'rgb', 'lab'), 'lab')

    for rc in reqColors:
        rgb = getRGBDists(rc, palettes)
        hsv = getHSVDists(rc, palettes)
        perc = getPerceptualDists(rc, palettes)

        # average across required colors
        for id, dist in rgb.items():
            if id not in rgbDists:
                rgbDists[id] = dist/len(reqColors)
            else:
                rgbDists[id] += dist/len(reqColors)

        for id, dist in hsv.items():
            if id not in hsvDists:
                hsvDists[id] = dist/len(reqColors)
            else:
                hsvDists[id] += dist/len(reqColors)

        for id, avg in perc.items():
            if id not in percDists:
                percDists[id] = avg/len(reqColors)
            else:
                percDists[id] += avg/len(reqColors)

    keywordAvgs = keyword(keywords, palettes)

    # weighted average of scores
    for id,palette in palettes.items():
        score = (1 - rgbDists[id]/maxRGB)*rgbW
        score += (1 - hsvDists[id]/maxHSV)*hsvW
        score += (1 - percDists[id]/maxPerc)*percW
        score += keywordAvgs[id]*keyW

        scoreDict[id] = (palette, score)

    return scoreDict












#
