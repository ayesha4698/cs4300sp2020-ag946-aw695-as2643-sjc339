from . import *
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
from app.irsystem.controllers.IR_main import *
from quickjs import Function
from colormap import rgb2hex
from PIL import ImageColor
import re
import requests
import csv
import json

project_name = "Version 1: Color Palette"
net_id = "Ayesha Gagguturi(ag946)"


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
    print(energy)

    rgb = convertColor(color, 'hex', 'rgb')
    print(rgb)

    # r = rgb[0]
    # g = rgb[1]
    # b = rgb[2]

    hsl = convertColor(color, 'hex', 'hsl')
    # print(hsl)

    # print(hsl[0])
    # print(hsl[1])
    # print(hsl[2])

    newBrightness = hsl[2] + hsl[2] * ((energy * 20) // 100)

    # print(newBrightness)

    new_rgb = str(hsl[0]) + ',' + str(hsl[1]) + ',' + str(newBrightness)
    # print(hsl[0])
    # print(hsl[1])
    # print(newBrightness)
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
    with open("Cymbolism.csv", newline="") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=",")
        word_dict = {}
        count = 0
        color_index = {}
        # for row in spamreader:
        #     if count == 0:
        #         index = 0
        #         for color in row:
        #             if color != 'word':
        #                 color_index[color] = index
        #                 index += 1
        #     else:
        #         word_dict[row[0]] = row[1:]

        #     count += 1

        for row in spamreader:
            # print('here')
            if count == 0:
                index = 0
                for color in row:
                    if color != 'word':
                        color_index[index + 1] = color
                        index += 1
            else:
                # print('here')
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
        # print(color_index)
        # print(word_dict)
    return word_dict


def create_combo_hex_codes(keywords, necessary_color):
    """ Calls the palette generator helper
:param keywords: list of words 
:param necessary_color: list of hex codes 
    :param 
:return: a list of lists of hex color codes
"""
    return 0


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
        # print(lst)
        sort = sorted(lst, reverse=True)

        sorted_word_dict[word] = sort
    # print(sorted_word_dict)

    for word in keywords:
        if word in sorted_word_dict:
            # print('here')
            tup = sorted_word_dict[word][0]
            color = tup[1]
            adj_color = energy_adjust(color, energy)
            top_colors.append(adj_color)

    # print(top_colors)
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
            input_lst.append(ImageColor.getcolor(hex, "RGB"))

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
        hex_codes = top_keywords_color_lst + necessary_color_lst
        combo_hex_code_lst.append(palette_generator(hex_codes, n))
    return combo_hex_code_lst


def backend_main(keywords, necessary_colors, energy, num_colors):
    """ compute the colors from the user input
:param keywords: list of words 
:param necessary_color: list of hex codes 
    :param energy: 
:return: dict with the format {palette_id: [list of hexcodes, ...],...}}
"""
    palette_dict = {}
    top_colors = top_colors_from_keywords(keywords, energy)

    palettes = create_combo_hex_codes([top_colors], necessary_colors)

    index = 0
    for p in palettes:
        palette_dict[index] = p
        index += 1


@irsystem.route("/", methods=["GET", "POST"])
def validate():
    keywords = ""
    energy = ""
    color1 = ""
    color2 = ""
    numcolors = ""

    errors = []
    results = ""

    if request.method == "POST":
        # keywords only singluar words, not phrases
        keywords = request.form["keywords"]
        energy = request.form["energy"]
        color1 = request.form["color1"]
        color2 = request.form["color2"]
        numcolors = request.form["numcolors"]

        print(keywords)
        print(energy)
        print(color1)
        print(color2)
        print(numcolors)

        results = "yay"

        if not keywords:
            errors.append("keywords")
        if keywords:
            keywords = keywords.replace(" ", "").split(",")
            if len(keywords) == 0:
                errors.append(keywords)
        if not energy:
            errors.append("energy")

        print(re.search("^([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$", color1))
        if color1 and re.search("^([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$", color1) == None:
            errors.append("color1")
        if color2 and re.search("^([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$", color2) == None:
            errors.append("color2")
        if not numcolors:
            errors.append("numcolors")

    if len(errors) == 0:
        return render_template('search.html', results=results, keywords=keywords, energy=energy, color1=color1, color2=color2, numcolors=numcolors)
    else:
        keywordString = ", ".join(map(str, keywords))
        print(keywordString)
        return render_template('search.html', errors=errors, keywords=keywordString, energy=energy, color1=color1, color2=color2, numcolors=numcolors)
