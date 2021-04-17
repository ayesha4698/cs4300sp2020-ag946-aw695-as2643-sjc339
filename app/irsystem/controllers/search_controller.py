from . import *  
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
import re

project_name = "Ilan's Cool Project Template"
net_id = "Ilan Filonenko: if56"

# @irsystem.route('/', methods=['GET'])
# def search():
# 	query = request.args.get('search')
# 	if not query:
# 		data = []
# 		output_message = ''
# 	else:
# 		output_message = "Your search: " + query
# 		data = range(5)
# 	return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data)

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
	#keywords only singluar words, not phrases
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

	





