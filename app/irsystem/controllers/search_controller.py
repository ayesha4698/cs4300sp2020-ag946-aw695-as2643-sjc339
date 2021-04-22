from . import *
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
# from app.irsystem.controllers.IR_main import *
from app.irsystem.controllers.IR_helpers import *
from app.irsystem.controllers.rgb2lab import *
from quickjs import Function
from colormap import rgb2hex
from PIL import ImageColor
import re
import requests
import csv
import json
import colorsys

netid = "Ayesha Gagguturi (ag946), Joy Thean (jct263), Skylar Capasso (sjc339), Anishka Singh (as2643), Alisa Wong (aw695)"

#### IR HELPERS #####
def deltaE(lab1, lab2):
    """
    Returns the perceptual distance between two colors in CIELAB.

    Params: lab1    color in LAB code [Tuple of Ints]
            lab2    color in LAB code [Tuple of Ints]
    """

    dL = lab1[0] - lab2[0]
    dA = lab1[1] - lab2[1]
    dB = lab1[2] - lab2[2]

    c1 = math.sqrt(lab1[1]**2 + lab1[2]**2)
    c2 = math.sqrt(lab2[1]**2 + lab2[2]**2)

    dC = c1 - c2
    dH = dA**2 + dB**2 - dC**2

    if dH < 0:
        dH = 0
    else:
        dH = math.sqrt(dH)

    sc = 1.0 + 0.045 * c1
    sh = 1.0 + 0.015 * c1

    dLKlsl = float(dL)
    dCkcsc = dC/sc
    dHkhsh = dH/sh

    dE = dLKlsl**2 + dCkcsc**2 + dHkhsh**2

    if dE < 0:
        return 0
    else:
        return math.sqrt(dE)


def colorDiff(c1, c2, code):
    """
    Returns the distance between two colors in either RGB space or HSL space.

    Params: c1      color in RGB, HSL, or LAB code [Tuple of Ints]
            c2      color in RGB, HSL, or LAB code [Tuple of Ints]
            code    'rgb', 'hsv' [String]
    """
    if code == 'rgb':
        return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2 + (c1[2] - c2[2])**2)

    elif code == 'hsv':
        diff = (math.sin(c1[0])*c1[1]*c1[2] - math.sin(c2[0])*c2[1]*c2[2])**2
        diff += (math.cos(c1[0])*c1[1]*c1[2] - math.cos(c2[0])*c2[1]*c2[2])**2
        diff += (c1[2] - c2[2])**2
        return diff*100

    elif code == 'lab':
        return deltaE(c1, c2)


def getRGBDists(reqColor, palettes):
    """
    Returns a dictionary where the keys are palette IDs and the values are
        the minimum RGB euclidian distances from the required color to every
        color on the palette.

    Params: reqColor    user-inputted clean hexcode color [String]
            palettes    palette IDs to Lists of clean hexcodes [Dict of Lists of Strings]
    """

    rgbDists = {}

    reqRGB = convertColor(reqColor, 'hex', 'rgb')

    for id, palette in palettes.items():
        minDist = 500
        for c in palette:
            cRGB = convertColor(c, 'hex', 'rgb')
            dist = colorDiff(reqRGB, cRGB, 'rgb')
            if dist < minDist:
                minDist = dist
        rgbDists[id] = minDist

    return rgbDists


def getHSVDists(reqColor, palettes):
    """
    Returns a dictionary where the keys are palette IDs and the values are
        the minimum HSV cartesian distances from the required color to every
        color on the palette.

    Params: reqColor    user-inputted clean hexcode color [String]
            palettes    palette IDs to Lists of clean hexcodes [Dict of Lists of Strings]
    """

    hsvDists = {}

    reqHSV = convertColor(reqColor, 'hex', 'hsl')
    reqHSV = convertColor(reqHSV, 'hsl', 'hsv')

    for id, palette in palettes.items():
        minDist = 500
        for c in palette:
            cHSV = convertColor(c, 'hex', 'hsl')
            cHSV = convertColor(cHSV, 'hsl', 'hsv')
            dist = colorDiff(reqHSV, cHSV, 'hsv')
            if dist < minDist:
                minDist = dist
        hsvDists[id] = minDist

    return hsvDists


def getPerceptualDists(reqColor, palettes):
    """
    Returns a dictionary where the keys are palette IDs and the values are
        the minimum perceptual distance of the required color to every
        color on the palette.

    Params: reqColor    user-inputted clean hexcode color [String]
            palettes    palette IDs to Lists of clean hexcodes [Dict of Lists of Strings]
    """

    deltaEDists = {}

    reqLAB = convertColor(reqColor, 'hex', 'rgb')
    reqLAB = convertColor(reqLAB, 'rgb', 'lab')

    for id, palette in palettes.items():
        minDist = 500
        for c in palette:
            cLAB = convertColor(c, 'hex', 'rgb')
            cLAB = convertColor(cLAB, 'rgb', 'lab')
            dist = colorDiff(reqLAB, cLAB, 'lab')
            if dist < minDist:
                minDist = dist
        deltaEDists[id] = minDist

    return deltaEDists


def CloseColorHelper(cymColors, colorToMatch):
    """
      Gets the closest color to the Cymbolism list of colors
      based on the RGB distance

      Params: cymColors: list of 19 colors from the Cymbolism website (hexcodes)  = Clean - without hashtag
              colorToMatch: one color from the palette to match

      Returns: one of the 19 colors ( hexcode)

      """

    returnlist = {}
    colorToMatch = convertColor(colorToMatch, 'hex', 'rgb')
    for x in range(len(cymColors)):
        rgbcolor = convertColor(cymColors[x], 'hex', 'rgb')
        distance = colorDiff(rgbcolor, colorToMatch, 'rgb')
        returnlist[cymColors[x]] = distance

    larg = 0
    ret = ""
    for k, v in returnlist.items():
        if v > larg:
            ret = k
    return ret


def keyword(userWords, paletteDict):
    """
      Returns a dictionary that includes the percentage score based on the colors and keywords

      Params: userWords: the keywords that the user inputted matched
      to a cymbolism words
              paletteDict: dictionary of the palettes
              data is dictionary where the key is the keyword, the value is list where each c

      Returns: Dictionary in format: {palette_id: average,...}
      """

    colordict = {}
    cymColors = list(cymColorsInvInd.keys())
    for palette in paletteDict.keys():
        score = 0
        for word in userWords:
            lst = []
            for color in paletteDict[palette]:
                closecolor = CloseColorHelper(cymColors, color)
                lst = cymData[word]
                ind = cymColorsInvInd[closecolor]
                colorScore = float(lst[ind])
            score += colorScore
        colordict[palette] = score
    return colordict

def convertColor(color, fromCode, toCode):
    """
    Returns a color converted from one code system to another. None if any
        params are incorrectly formatted.

    Ex: convertColor('(255,255,255)', 'rgb', 'hex') -> 'FFFFFF'

    Params: color       clean hexcode, rgb, hsl [String or Tuple of Ints]
            fromCode    'hex', 'rgb', 'hsl' [String]
            toCode      'hex', 'rgb', 'hsl', 'hsv', 'lab' [String]
    """
    if type(color) == str and "#" in color:
        color = clean_hex(color)

    if fromCode == 'hsl' and toCode == 'hsv':
        v = color[2]/100 + color[1]/100*min(color[2]/100, 1-color[2]/100)
        if v == 0:
            s = 0
        else:
            s = 2*(1 - color[2]/100/v)
        return (color[0], s, v)

    elif fromCode == 'rgb' and toCode == 'lab':
        return tuple(rgb2lab(color))

    elif fromCode == 'rgb' and toCode == 'hsv':
        return colorsys.rgb_to_hsv(color[0], color[1], color[2])

    elif fromCode == 'rgb' and toCode == 'hsl':
        newcolor = colorsys.rgb_to_hsv(color[0], color[1], color[2])
        return (newcolor[0], newcolor[2], newcolor[1])

    elif fromCode == 'rgb' and toCode == 'hex':
        return '%02x%02x%02x' % color

    elif fromCode == 'hex' and toCode == 'rgb':
        return tuple(int(color[i:i+2], 16) for i in (0, 2, 4))

    elif fromCode == 'hex' and toCode == 'hsl':
        rgb = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
        newcolor = colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2])
        return (newcolor[0], newcolor[2], newcolor[1])

    else:
        raise ValueError('Invalid inputs to convertColor: ' + str(color) + ', '
            + fromCode + ', ' + toCode)

# def convertColor(color, fromCode, toCode):
#     """
#     Returns a color converted from one code system to another. None if any
#         params are incorrectly formatted.

#     Ex: convertColor('(255,255,255)', 'rgb', 'hex') -> 'FFFFFF'

#     Params: color       clean hexcode, rgb, hsl [String or Tuple of Ints]
#             fromCode    'hex', 'rgb', 'hsl' [String]
#             toCode      'hex', 'rgb', 'hsl', 'hsv', 'lab' [String]
#     """
#     if type(color) == str and "#" in color:
#         color = clean_hex(color)

#     if fromCode == 'hsl' and toCode == 'hsv':
#         v = color[2]/100 + color[1]/100*min(color[2]/100, 1-color[2]/100)
#         if v == 0:
#             s = 0
#         else:
#             s = 2*(1 - color[2]/100/v)
#         return (color[0], s, v)

#     elif fromCode == 'rgb' and toCode == 'lab':
#         return tuple(rgb2lab(color))

#     query = '/id?' + fromCode + '=' + color
#     url = 'https://www.thecolorapi.com' + query + '&format=json'

#     context = ssl._create_unverified_context()
#     response = urllib.request.urlopen(url, context=context)
#     color_json = json.loads(response.read().decode())

#     if None in color_json['rgb'].values():
#         query = '/id?' + fromCode + '=' + color_json['hex']['clean']
#         url = 'https://www.thecolorapi.com' + query + '&format=json'

#         context = ssl._create_unverified_context()
#         response = urllib.request.urlopen(url, context=context)
#         color_json = json.loads(response.read().decode())

#     if toCode == 'hex':
#         return color_json['hex']['clean']
#     elif toCode == 'rgb':
#         return (int(color_json['rgb']['r']), int(color_json['rgb']['g']),
#                 int(color_json['rgb']['b']))
#     elif toCode == 'hsl':
#         return (int(color_json['hsl']['h']), int(color_json['hsl']['s']),
#                 int(color_json['hsl']['l']))
#     else:
#         return None

### END IR HELPERS ####

def hue_adjuster():
    return 0


def energy_adjust(color, energy):
    """ adjusts a hex code so that it’s brighter or more muted
    :param hex: string
    :param brightness: decimal for the percentage of how bright
    Input - 5 * 2 * 10 (test how large increase should be)
    Return: hex code
"""

    energy -= 5
    # print(energy)

    rgb = convertColor(color, 'hex', 'rgb')
    # print(rgb)

    hsl = convertColor(color, 'hex', 'hsl')

    newBrightness = hsl[2] + hsl[2] * ((energy * 20) // 100)

    # print(newBrightness)

    new_rgb = str(hsl[0]) + ',' + str(hsl[1]) + ',' + str(newBrightness)

    updated_color = convertColor(new_rgb, 'rgb', 'hex')
    return updated_color


def clean_hex(color):
    """

      This function converts the color names in the dataset to just be a hex number
      For example:
        'olive #808000' -> 808000

      Params: color     string of form 'color #hex'

    """
    split = color.find('#')
    return color[split+1:]


def parse_data():
    """
      This function parses the dataset. It returns a dictionary in which the
      keys are the words from Cymbolism. The values of each key is a list
      of tuples (unsorted) where the second element of the tuple is a color
      and the first element is the score of that color.
    """
    with open("data/Cymbolism.csv", newline="") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=",")
        word_dict = {}
        count = 0
        color_index = {}

        for row in spamreader:

            if count == 0:
                index = 0
                for color in row:
                    if color != 'word':
                        color_index[index + 1] = color
                        index += 1
            else:

                index = 0
                word = ''
                for value in row:
                    if index == 0:
                        word = row[0]
                    else:
                        if word in word_dict:
                            word_dict[word].append(
                                (float(row[index]), clean_hex(color_index[index])))
                        else:
                            word_dict[word] = [
                                (float(row[index]), clean_hex(color_index[index]))]
                    index += 1

            count += 1

    return word_dict


def top_colors_from_keywords(keywords, energy):
    """ Generating each keywords top color from the dataset
:param keywords: list of words
    :param energy: string of energy level
:return: list of top colors with energy adjusted (hex codes)
"""
    top_colors = []

    word_dict = parse_data()
    sorted_word_dict = {}

    for word, lst in word_dict.items():
        sort = sorted(lst, reverse=True)

        sorted_word_dict[word] = sort

    for word in keywords:
        if word in sorted_word_dict:
            tup = sorted_word_dict[word][0]
            color = tup[1]
            # adj_color = energy_adjust(color, energy)
            # top_colors.append(adj_color)
            top_colors.append(color)

    return top_colors


def palette_generator(hex_codes, n):
    """ Adds on the “N” values to the list of input colors and outputs the api generated palette
    need to convert inputs to rgb and outputs to hex
:param hex_codes: list of hex colors (up to three)
    :param n: number of colors
:return: list of hex color codes computed from the Colormind API
"""

    url = "http://colormind.io/api/"

    input_lst = []
    if n <= 3:
        for hex in hex_codes:
            input_lst.append(convertColor(hex, "hex", "rgb"))
            # input_lst.append(ImageColor.getcolor(hex, "RGB"))

    for add_color in range(n, 5):
        input_lst.append("N")

    data = {
        "model": "default",
        "input": input_lst
    }
    res = requests.post(url, data=json.dumps(data))

    hex = []
    for rgb in res.json()['result']:
        hex.append(rgb2hex(rgb[0], rgb[1], rgb[2]))

    return hex


def create_combo_hex_codes(top_keywords_color_lst, necessary_color_lst):
    """ Calls the palette generator helper
:param keywords: list of list top keywords colors
:param necessary_color: list of list hex codes
    :param
:return: a list of lists of hex color codes
"""
    combo_hex_code_lst = []
    for i in range(len(top_keywords_color_lst)):
        n = len(top_keywords_color_lst)+len(necessary_color_lst)
        hex_codes = top_keywords_color_lst[i] + necessary_color_lst
        combo_hex_code_lst.append(palette_generator(hex_codes, n))
    return combo_hex_code_lst


def input_to_color(keywords, necessary_colors, energy, num_colors):
    """ compute the colors from the user input
:param keywords: list of words
:param necessary_color: list of hex codes
    :param energy:
:return: dict with the format {palette_id: [list of hexcodes, ...],...}}
"""
    energy = int(energy)
    num_colors = int(num_colors)
    palette_dict = {}
    top_colors = top_colors_from_keywords(keywords, energy)
    print("top colors")
    print(top_colors)

    palettes = create_combo_hex_codes([top_colors], necessary_colors)

    index = 0
    for p in palettes:
        palette_dict[index] = p
        index += 1

    return palette_dict


#### IR MAIN!!!!!!!!!!!!! ######

# dataset globals
cymColorsInvInd = {}
cymData = {}
cymVotes = {}

# /Users/ayesha/cs4300sp2020-ag946-aw695-as2643-sjc339/app/irsystem/controllers/IR_main.py
with open('data/Cymbolism.csv', mode='r') as infile:
    reader = csv.reader(infile)
    for rows in reader:
        cymData[rows[0]] = rows[1:-1]
        cymVotes[rows[0]] = rows[-1]

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
        score = 0
        if (rgbDists != {}):
            score += (1 - rgbDists[id]/maxRGB)*rgbW
        if (hsvDists != {}):
            score += (1 - hsvDists[id]/maxHSV)*hsvW
        if (percDists != {}):
            score += (1 - percDists[id]/maxPerc)*percW
        if (keywordAvgs != {}):
            score += keywordAvgs[id]*keyW

        scoreDict[id] = (palette, score)

    return scoreDict


@irsystem.route("/", methods=["GET"])
def search():
    valid = True

    keywords = request.args.get("keywords")
    energy = request.args.get("energy")
    color1 = request.args.get("color1")
    color2 = request.args.get("color2")
    numcolors = request.args.get("numcolors")

    keywordString = ""
    errors = []

    submit = True
    if (keywords == None and energy == None and color1 == None and color2 == None and numcolors == None):
        submit = False

    print(keywords)
    print(energy)
    print(color1)
    print(color2)
    print(numcolors)

    if not keywords:
        errors.append("keywords")
    if keywords:
        keywords = keywords.replace(" ", "").split(",")
        if len(keywords) == 0:
            errors.append("keywords")
        else:
            keywordString = ",".join(map(str, keywords))

    if not energy:
        errors.append("energy")

    if color1 and re.search("^([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$", color1) == None:
        errors.append("color1")
    if color2 and re.search("^([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$", color2) == None:
        errors.append("color2")

    if not numcolors:
        errors.append("numcolors")

    reqColors = []
    if color1:
        reqColors.append(color1)
    if color2:
        reqColors.append(color2)

    results = ""
    if len(errors) == 0:
        results = getPalettes(keywords, reqColors, energy, numcolors)

    return render_template('search.html', netid=netid, results=results, keywords=keywordString, energy=energy, color1=color1, color2=color2, numcolors=numcolors, errors=errors, submit=submit)
