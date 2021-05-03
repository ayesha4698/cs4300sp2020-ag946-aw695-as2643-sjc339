from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import time
import math
from . import *
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder

# from app.irsystem.controllers.IR_main import *
# from app.irsystem.controllers.IR_helpers import *
from app.irsystem.controllers.rgb2lab import *
from app.irsystem.controllers.cossim import *

from colormap import rgb2hex
import re
import requests
import csv
import json
import colorsys
import nltk
import itertools
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

###########################################################################################
#                                          GLOBALS                                        #
###########################################################################################

# dataset globals
cymData = {}
cymColorsInvInd = {}
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

stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")

netid = "Alisa Wong (aw695), Anishka Singh (as2643), Ayesha Gagguturi (ag946), Joy Thean (jct263), and Skylar Capasso (sjc339)"


###########################################################################################
#                                       IR HELPERS                                        #
###########################################################################################

def to_wordnet(tag):
    """
    Returns a wordnet PartOfSpeech object. None if no valid wordnet tag.
    Params: tag     wordnet PartOfSpeech string
    """
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    elif tag.startswith('V'):
        return wordnet.VERB
    return None


def searchCymDfns(w1):
    """
    Returns the similarity score and Cymbolism word match for a given Synset object.
    This function searches all the definitions of all the Cymbolism words to
    find the highest similarity score with the given keyword.

    Params: w1      Synset object
    """
    match = ''
    maxSim = 0
    # dummy run for speed
    # arbitrary word that doesn't have match in dataset
    wt = wordnet.synsets('cool')[-1]
    for cymW in cymData.keys():
        if cymW != 'word':
            cymW = cymW.replace(" ", "_")
            for w2 in wordnet.synsets(cymW):
                sim = wt.wup_similarity(w2)
                if sim is None:
                    sim = wt.path_similarity(w2)
                if sim is not None and sim > maxSim:
                    maxSim = sim
                    match = cymW

    # real run
    for cymW in cymData.keys():
        if cymW != 'word':
            cymW = cymW.replace(" ", "_")
            for w2 in wordnet.synsets(cymW):
                sim = w1.wup_similarity(w2)
                if sim is None:
                    sim = w1.path_similarity(w2)
                if sim is not None and sim > maxSim:
                    maxSim = sim
                    match = cymW

    # maxSim = max([0, maxSim - .1])                  # adjust? fix later
    return maxSim, match


def searchKeywordDfn(w1):
    """
    Returns the similarity score and Cymbolism word match for a given Synset object.
    This function searches all the definitions of all the Cymbolism words to
    find the highest similarity score with the given keyword.

    Params: w1      Synset object
    """
    match = ''
    maxSim = 0
    toks = tokenize(w1.definition())
    toks = [w for w in toks if not w in stop_words]

    tagged = nltk.pos_tag(toks)
    lemmatzr = WordNetLemmatizer()

    synsets = []
    n = []
    v = []
    a = []
    r = []
    s = []

    for tok in tagged:
        wn_tag = to_wordnet(tok[1])
        if not wn_tag:
            continue

        lemma = lemmatzr.lemmatize(tok[0], pos=wn_tag)
        if wordnet.synsets(lemma, pos=wn_tag) != []:
            word = wordnet.synsets(lemma, pos=wn_tag)[0]
        else:
            word = wordnet.synsets(lemma)[0]

        if word.pos() == 'n':
            n = n + [word]
        elif word.pos() == 'v':
            v = v + [word]
        elif word.pos() == 'a':
            a = a + [word]
        elif word.pos() == 'r':
            r = r + [word]
        elif word.pos() == 's':
            s = s + [word]
        else:
            synsets = synsets + [word]

    synsets = a + n + s + r + v + synsets

    for w3 in synsets:
        for cymW in cymData.keys():
            if cymW != 'word':
                cymW = cymW.replace(" ", "_")
                for w2 in wordnet.synsets(cymW):
                    sim = w3.wup_similarity(w2)
                    if sim is None:
                        sim = w3.path_similarity(w2)
                    if sim is not None and sim > maxSim:
                        maxSim = sim
                        match = cymW

    # maxSim = max([0, maxSim - .2])                  # adjust? fix later
    return maxSim, match


def getSynset(dfn):
    """
    Returns the Synset object that matches the word and
        definition inputted by the user.

    Params: dfn     User's keyword in the format:
                        'word - definition' [String]
    """
    #print('DFN_-----------------')
    #print(dfn)
    #print(dfn.find("-"))
    if not dfn.find("-") == -1:
        kw = dfn[:dfn.index("-")-1]         # skip the space

        kw = stemmer.stem(kw)
        #print(kw)
        syns = wordnet.synsets(kw.replace(" ", "_"))
        #print(syns, kw)
        if syns == []:
            kw = dfn[:dfn.index("-")-1]
            syns = wordnet.synsets(kw.replace(" ", "_"))

        query = dfn[dfn.index("-")+2:]      # skip the space
    else:
        return wordnet.synsets(dfn)[0]

    # find Synset object based on definition
    msgs = []
    for syn in syns:
        toks = tokenize(syn.definition())
        msgs.append({'toks': toks})

    inv_idx = build_inverted_index(msgs)
    idf = compute_idf(inv_idx, len(msgs))
    inv_idx = {key: val for key, val in inv_idx.items()
               if key in idf}
    doc_norms = compute_doc_norms(inv_idx, idf, len(msgs))
    ind_search = index_search(query, inv_idx, idf, doc_norms)

    msg_id = ind_search[0][1]
    return syns[msg_id]


def keywordMatch(dfns):
    """
    Returns a list of Cymbolism keywords that match to each keyword of the user's
        input, of the following format:
        [(keyword, similarity score),
        ...
        (keyword, similarity score)]
    Params: dfns    List of user's keywords where each string is formatted:
                        'word - definition' [List of Strings]
    """
    synwords = []
    keywords = []

    for dfn in dfns:
        if dfn in cymData.keys() and dfn != 'word':
            synwords.append(wordnet.synsets(dfn)[0])
        else:
            s = getSynset(dfn)
            # if s is not None:
            synwords.append(s)
    #print('syn', synwords)
    for w1 in synwords:
        match = ''
        maxSim = 0

        # keyword or keyword's synonyms are in Cymbolism
        if maxSim == 0:
            for lem in w1.lemmas():
                if lem.name() in cymData.keys() and lem.name() != 'word':
                    maxSim = 1.0
                    match = lem.name()
                    break

        # keyword matches Cymbolism word's meanings
        if maxSim == 0:
            maxSim, match = searchCymDfns(w1)

        # keyword's definition matches Cymbolism word's meanings
        if maxSim == 0:
            maxSim, match = searchKeywordDfn(w1)

        keywords.append((match, maxSim))

    return keywords


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

    #print("RGBDIST")
    #print(palettes)
    #print(reqColor)

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

    small = 1000
    ret = ""
    for k, v in returnlist.items():
        if v < small:
            small = v
            ret = k
    return ret


def keyword(userWords, paletteDict):
    """
      Returns one dictionary that includes the percentage score based on the colors and keywords
      Returns another dictionary that also includes the keyword and the corresponding percentage score
      for each palette
      Params: userWords: the keywords that the user inputted matched
      to a cymbolism words
              paletteDict: dictionary of the palettes
              data is dictionary where the key is the keyword, the value is list where each c
      Returns: Dictionary in format: {palette_id: average,...}
                                    {palette_id: [(keyword, percent), (keyword, percent),...],...}
      """

    colordict = {}
    keywordDict = {}
    cymColors = list(cymColorsInvInd.keys())
    maxScore = 0
    for palette in paletteDict.keys():
        score = 0
        keywordDict[palette] = []
        for word in userWords:
            lst = []
            for color in paletteDict[palette]:
                closecolor = CloseColorHelper(cymColors, color)
                lst = cymData[word]
                ind = cymColorsInvInd[closecolor]
                colorScore = float(lst[ind])
                # print()
                # print("COLOR")
                # print(word)
                #print(color)
                #print(closecolor)
                # print(colorScore)
            score += colorScore
            keywordDict[palette].append((word, colorScore))
        if score > maxScore:
            maxScore = score
        colordict[palette] = score

    for id in colordict:
        colordict[id] /= maxScore

    return colordict, keywordDict


def convertColor(color, fromCode, toCode):
    """
    Returns a color converted from one code system to another. None if any
        params are incorrectly formatted. (all clean hexcodes)
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

    rgb = convertColor(color, 'hex', 'rgb')

    r = rgb[0]
    g = rgb[1]
    b = rgb[2]

    if energy > 0:
        energy *= 10
        newR = r + (256 - r) * energy // 100
        newG = g + (256 - g) * energy // 100
        newB = b + (256 - b) * energy // 100

        newRGB = (abs(newR), abs(newG), abs(newB))
    elif energy < 0:
        energy /= 10
        newR = round(min(max(0, r + (r * energy)), 255))
        newG = round(min(max(0, g + (g * energy)), 255))
        newB = round(min(max(0, b + (b * energy)), 255))

        newRGB = (abs(newR), abs(newG), abs(newB))
    else:
        newRGB = rgb

    newHex = convertColor(newRGB, 'rgb', 'hex')
    #print('NEWHEX')
    #print(newHex)
    newHex = newHex.upper()
    #print(newHex)
    return newHex


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

        total_votes = {}
        for row in spamreader:

            votes_index = len(row) - 1
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
                    elif index == votes_index:
                        total_votes[word] = float(value)
                    else:
                        if word in word_dict:
                            word_dict[word].append(
                                (float(row[index]), clean_hex(color_index[index])))
                        else:
                            word_dict[word] = [
                                (float(row[index]), clean_hex(color_index[index]))]
                    index += 1

            count += 1

    return word_dict, total_votes


def normalize_score(score, num_votes, total_votes):
    """
        :param score: float
        :param necessary_color: list of hex codes
        :param energy:
        :return: normalized score (float)
    """
    # #print((score * num_votes) / total_votes)
    return (score * num_votes) / total_votes


def top_colors_from_keywords(keywords, energy):
    """ Generating each keywords top color from the dataset
        :param keywords: tuple (keyword (str), similarity score (float))
        :param energy: string of energy level
        :return: list tuples (top colors with energy adjusted (hex codes), score of color)
    """
    top_colors = {}

    word_dict, total_votes = parse_data()
    sorted_word_dict = {}

    for word, lst in word_dict.items():
        sort = sorted(lst, reverse=True)
        sorted_word_dict[word] = sort

    total_votes_num = 0
    for tup in keywords:
        word = tup[0]
        #print("----------WORDS")
        #print(keywords)
        # #print(total_votes)
        #print(word)
        total_votes_num += total_votes[word]

    for tup in keywords:
        word = tup[0]
        sim_score = tup[1]
        if word in sorted_word_dict:
            above_thresh = True
            while above_thresh:
                tups = sorted_word_dict[word]
                for tup in tups:
                    score = tup[0]
                    color = tup[1]
                    if score >= 10:
                        if color in top_colors:
                            top_colors[color] += sim_score * normalize_score(
                                (score), total_votes[word], total_votes_num)
                        else:
                            top_colors[color, energy] = sim_score * normalize_score(
                                (score), total_votes[word], total_votes_num)
                    else:
                        above_thresh = False

    return top_colors

# top_colors_from_keywords([('beach', 1), ('beauty', 1)], 5)
# top_colors_from_keywords([('baby', 0.5), ('beach', 1)], 5)
# top_colors_from_keywords([('abuse', 1)], 5)
# top_colors_from_keywords([('cold', 1), ('snorkel', 1)], 5)


def palette_generator(hex_codes, n, energy):
    """ Adds on the “N” values to the list of input colors and outputs the api generated palette
    need to convert inputs to rgb and outputs to hex
:param hex_codes: list of hex colors (up to three)
    :param n: number of colors
:return: list of hex color codes computed from the Colormind API
"""

    #print("HEX CODES")
    #print(hex_codes)

    url = "http://colormind.io/api/"

    input_lst = []
    for hex in hex_codes:
        #print(hex)
        input_lst.append(convertColor(hex[0], "hex", "rgb"))
        # input_lst.append(ImageColor.getcolor(hex, "RGB"))
    n = len(input_lst)
    for add_color in range(n, 5):
        input_lst.append("N")

    #print("INPUT LIST")
    #print(input_lst)
    data = {
        "model": "default",
        "input": input_lst
    }
    #print('DATA')
    #print(data)
    res = requests.post(url, data=json.dumps(data))

    hex = []
    for rgb in res.json()['result']:
        color = rgb2hex(rgb[0], rgb[1], rgb[2])
        hex.append(energy_adjust(color, energy))

    #print('HEX')
    #print(hex)
    return hex


def create_combo_hex_codes(top_keywords_color_lst, necessary_color_lst, energy):
    """ Calls the palette generator helper
:param keywords: list of list top keywords colors
:param necessary_color: list of list hex codes
    :param
:return: a list of lists of hex color codes
"""
    combo_hex_code_lst = []

    #print('necc', necessary_color_lst)

    necessary_colors = []

    for c in necessary_color_lst:
        necessary_colors.append((c, 100))


    keyword_lst = list(set(top_keywords_color_lst.keys()))

    # start = time.time()
    combinations_object = list(itertools.combinations(keyword_lst, 2))
    #print('combinations time')
    #print(time.time() - start)

    # combo_short = combinations_object[:5]

    if(len(combinations_object) > 100):
        combinations_object = combinations_object[:100]

    #print(len(combinations_object))

    for i in range(len(combinations_object)):
        n = len(combinations_object)+len(necessary_color_lst)
        hex_codes = necessary_colors + list(combinations_object[i])

        # start = time.time()
        #print("HEX")
        #print(hex_codes)

        combo_hex_code_lst.append(palette_generator(hex_codes, n, energy))
        #print('palette generator time')
        #print(time.time() - start)
    # for i in range(len(combo_short)):
    #     n = len(combo_short)+len(necessary_color_lst)
    #     hex_codes = necessary_color_lst + list(combo_short[i])

    #     start = time.time()

    #     combo_hex_code_lst.append(palette_generator(hex_codes, n))
    #     #print('palette generator time')
    #     #print(time.time() - start)

    return combo_hex_code_lst


def input_to_color(keywords, necessary_colors, energy):
    """ compute the colors from the user input
:param keywords: list of words
:param necessary_color: list of hex codes
    :param energy:
:return: dict with the format {palette_id: [list of hexcodes, ...],...}}
"""
    energy = int(energy)
    # num_colors = int(num_colors)
    palette_dict = {}
    # #print('keywords', keywords)

    # start = time.time()
    top_colors = top_colors_from_keywords(keywords, energy)
    #print('TOP COLORS')
    #print(top_colors)
    # #print('top colors time')
    # #print(time.time() - start)

    # start = time.time()
    palettes = create_combo_hex_codes(top_colors, necessary_colors, energy)
    #print('PALETTES')
    #print(palettes)
    # #print('combo hex codes time')
    # #print(time.time() - start)

    # palettes is a list of lists - - so loop and change for necesarrycolors
    # #print("LOOK HERE ----------")
    # #print(palettes)
    for x in range(len(palettes)):
        for y in range(len(necessary_colors)):
            palettes[x][y] = necessary_colors[y]
    # #print()
    #print(palettes)
    index = 0
    for p in palettes:
        palette_dict[index] = p
        index += 1

    return palette_dict, top_colors


def getPalettes(keywords, reqColors, energy):
    """
    Returns a list of palettes sorted from highest to lowest ranked.
    Returns a dictionary of the following format: (all clean hexcodes)
        { paletteID: (List of Colors in Palette, Score),
          ...
          paletteID: (List of Colors in Palette, Score) }
    Returns another dictionary that also includes the keyword and the corresponding percentage score
      for each palette

    Params: keywords    List of user's keywords where each string is formatted:
                            'word - definition' [List of Strings]
            reqColors   list of user-inputted clean hexcode color [List of String]
            energy      user-input on the muted to bright scale [Int]
            numColors   number of colors the user wants in their palette [Int]
    """
    # start = time.time()
    # #print("key", keywords)
    cymKeywords = keywordMatch(keywords)
    # #print(cymKeywords)
    # print('keyword match time')
    #print(time.time() - start)
    # start = time.time()
    #print(cymKeywords)

    palettes, top_colors = input_to_color(cymKeywords, reqColors, energy)
    #print('input to color time')
    #print(time.time() - start)
    keywords = [i[0] for i in cymKeywords]

    # start = time.time()
    #print("SCORE PALETTES ")
    #print(keywords)
    #print(palettes)
    scored, keywordBreakdown = scorePalettes(palettes, keywords, reqColors, top_colors)

    #print('score palettes time')
    #print(time.time() - start)

    #print()
    #print(scored)

    ranked = sorted(scored.items(), key=lambda scored: scored[1][1], reverse=True)

    print()
    print(ranked)
    print()

    sortedScored = []
    i = 0
    while len(sortedScored) < 5 and i < len(ranked):
        tup = ranked[i]
        if tup[1][0] is not None:   # TODO: check if similar palette is already in sortedScored
            sortedScored.append(tup)
        i += 1

    print(sortedScored)

    return sortedScored, keywordBreakdown


def scorePalettes(palettes, keywords, reqColors, top_colors):
    """
    Returns a new dictionary that scores and ranks each palette based on the
        following factors and weights:
        1. RGB Euclidian Distance to Required Color                        25%
        2. HSV Cartesian Distance to Required Color                        25%
        3. Delta-E Distance to Required Color?                             0%
        4. Cymbolism Keyword Close Color Percentages                       50%
    Returns another dictionary that also includes the keyword and the corresponding percentage score
      for each palette
    Final dictionary will be of the following format: (all clean hexcodes)
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
    rgbW = .2
    hsvW = .2
    percW = .2
    keyW = .4

    # maximums
    # maxRGB = colorDiff((0, 0, 0), (255, 255, 255), 'rgb')
    # maxHSV = colorDiff((0, 0, 0), (0, 0, 100), 'hsv')
    # maxPerc = colorDiff(convertColor((0, 0, 0), 'rgb', 'lab'),
    #                     convertColor((255, 255, 255), 'rgb', 'lab'), 'lab')
    maxRGB = 0
    maxHSV = 0
    maxPerc = 0

    for rc in top_colors:
        rc = rc[0]
        rgb = getRGBDists(rc, palettes)
        hsv = getHSVDists(rc, palettes)
        perc = getPerceptualDists(rc, palettes)

        # average across top colors
        for id, dist in rgb.items():
            if dist > maxRGB:
                maxRGB = dist
            if id not in rgbDists:
                rgbDists[id] = dist/len(top_colors)
            else:
                rgbDists[id] += dist/len(top_colors)

        for id, dist in hsv.items():
            if dist > maxHSV:
                maxHSV = dist
            if id not in hsvDists:
                hsvDists[id] = dist/len(top_colors)
            else:
                hsvDists[id] += dist/len(top_colors)

        for id, avg in perc.items():
            if avg > maxPerc:
                maxPerc = avg
            if id not in percDists:
                percDists[id] = avg/len(top_colors)
            else:
                percDists[id] += avg/len(top_colors)

    keywordAvgs, keywordBreakdown = keyword(keywords, palettes)

    # weighted average of scores
    for id, palette in palettes.items():
        score = 0
        if (rgbDists != {}):
            score += (1 - rgbDists[id]/maxRGB)*100*rgbW
        if (hsvDists != {}):
            score += (1 - hsvDists[id]/maxHSV)*100*hsvW
        if (percDists != {}):
            score += (1 - percDists[id]/maxPerc)*100*percW
        if (keywordAvgs != {}):
            score += keywordAvgs[id]*100*keyW

        # TODO: uncomment this later when updown works
        # score += (100 - score)*(1 + net_updown/total_updown)

        scoreDict[id] = (palette, score)

    # print('scoredict')
    # print(scoreDict)

    return scoreDict, keywordBreakdown


@irsystem.route("/", methods=["GET", "POST"])
def search():
    if request.method == "POST":
        errors = []
        keywords = request.form.get("keywords")
        keywordDefs = []
        energy = request.form.get("energy")
        color1 = request.form.get("color1")
        color2 = request.form.get("color2")
        multiDefs = request.form.get("multiDefs")
        invalidWords = request.form.get("invalidWords")
        submit = True
        results = None
        showModal = False

        # print('HERE')
        # print(invalidWords)
        #print(keywords)
        #print(multiDefs)

        if (keywords is None and energy is None and color1 is None and color2 is None):
            submit = False

        if len(keywords) == 0 or keywords is None:
            errors.append("keywords1")
        if keywords:
            keywordsList = keywords.split(",")
            if len(keywordsList) == 0:
                errors.append("keywords1")
            elif len(keywordsList) > 5:
                errors.append("keywords2")
            elif invalidWords:
                invalidWordsList = invalidWords.split(",")
                for w in invalidWordsList:
                    if w not in keywordsList:
                        invalidWordsList.remove(w)

                if len(invalidWordsList) > 0:
                    #print("hello?????")
                    errors.append("keywords3")

        if color1 and re.search("^([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$", color1) is None:
            errors.append("color1")
        if color2 and re.search("^([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$", color2) is None:
            errors.append("color2")

        multiDefList = []

        if request.form.get('submit-button') == 'definitions':
            multiDefList = multiDefs.split(",")
            for d in multiDefList:
                print(d)
                try:
                    if request.form[d]:
                        keywordDefs.append(
                            d + " - " + request.form[d].replace("%", " "))
                except:
                    print("failed to get form element")
                    print(submit)
                    errors.append("multi")
                    return render_template('search.html', netid=netid, results=None, keywords=keywords, energy=energy, color1=color1, color2=color2, errors=errors, submit=submit, multiDefs=multiDefs, showModal=True)

        if request.form.get('submit-button') == 'general' and multiDefs:
            return render_template('search.html', netid=netid, results=None, keywords=keywords, energy=energy, color1=color1, color2=color2, errors=errors, submit=submit, multiDefs=multiDefs, showModal=True)

        for k in keywords.split(","):
            if k not in multiDefList:
                keywordDefs.append(k)

        reqColors = []
        if color1:
            reqColors.append(color1)
        if color2:
            reqColors.append(color2)

        # display results if no errors
        sortedScored = []
        keywordBreakdown = {}
        if len(errors) == 0:
            sortedScored, keywordBreakdown = getPalettes(
                keywordDefs, reqColors, energy)
            return render_template('search.html', netid=netid, sortedScored = sortedScored, keywordBreakdown=keywordBreakdown, keywordDefs=keywordDefs, keywords=keywords, energy=energy, color1=color1, color2=color2, errors=errors, submit=submit)

        # display errors + sticky values
        return render_template('search.html', netid=netid, sortedScored=None, keywords=keywords, energy=energy, color1=color1, color2=color2, errors=errors, submit=submit)

    return render_template('search.html', netid=netid)
    # return render_template('search.html', netid=netid, sortedScored=[(5, (['0FF4F3', 'F6DA0D', 'DDC114', 'C44D2A', 'BD2456'], 9.116932314670592)), (3, (['F9B00C', '0CF1F0', '79E6A2', 'A17B1E', 'D51F36'], 9.10481916108276)), (1, (['FDF606', '08FAF8', 'BBD4E5', '6E3F55', '16121F'], 1.6377623563612191)), (4, (['FBA10C', 'FDDA0B', 'AEA417', '153F2F', '131D29'], 1.5999848744714602)), (0, (['FCFA0A', 'FBA50C', 'E8321C', '1A2124', '15222E'], 1.5962150584780885))], keywordBreakdown={0: [('beach', 2.6857654431513)], 1: [('beach', 2.6857654431513)], 2: [('beach', 2.6857654431513)], 3: [('beach', 21.366756192181)], 4: [('beach', 2.6857654431513)], 5: [('beach', 21.366756192181)]}, keywords="beach", keywordDefs={"beach": None, "cool": "to lose heat"})
