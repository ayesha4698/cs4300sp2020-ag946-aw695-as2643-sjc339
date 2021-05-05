var hexRegex = "^([A-Fa-f0-9]{6})$";
var $select;
// let colorPicker1;
// let colorPicker2;

$(document).ready(function () {
    var cymbolism = ['abuse', 'accessible', 'addiction', 'agile', 'amusing', 'anger', 'anticipation', 'art deco', 'authentic', 'authority', 'average', 'baby', 'beach', 'beauty', 'beer', 'benign', 'bitter', 'blend', 'blissful', 'bold', 'book', 'boss', 'brooklyn', 'busy', 'calming', 'capable', 'car', 'cat', 'certain', 'charity', 'cheerful', 'chicago', 'classic', 'classy', 'clean', 'cold', 'colonial', 'comfort', 'commerce', 'compelling', 'competent', 'confident', 'consequence', 'conservative', 'contemporary', 'cookie', 'corporate', 'cottage', 'crass', 'creative', 'cute', 'dance', 'dangerous', 'decadent', 'decisive', 'deep', 'devil', 'discount', 'disgust', 'dismal', 'dog', 'drunk', 'dublin', 'duty', 'dynamic', 'earthy', 'easy', 'eclectic', 'efficient', 'elegant', 'elite', 'enduring', 'energetic', 'entrepreneur', 'environmental', 'erotic', 'excited', 'expensive', 'experience', 'fall', 'familiar', 'fast', 'fear', 'female', 'football', 'freedom', 'fresh', 'friendly', 'fun', 'furniture', 'future', 'gay', 'generic', 'georgian', 'gloomy', 'god', 'good', 'goth', 'government', 'grace', 'great', 'grow', 'happy', 'hard', 'hate', 'hazardous', 'hippie', 'hockey', 'honor', 'hope', 'hot', 'hunting', 'hurt', 'hygienic', 'ignorant', 'imagination', 'impossible', 'improbable', 'influence', 'influential', 'insecure', 'inviting', 'invulnerable', 'jacobean', 'jealous', 'joy', 'jubilant', 'junkie', 'knowledge', 'kudos', 'launch', 'lazy', 'leader', 'liberal', 'library', 'light', 'likely', 'lonely', 'love', 'magic', 'marriage', 'maximal', 'mean', 'medicine', 'melancholy', 'mellow', 'minimal', 'mission', 'modern', 'moment', 'money', 'music', 'mystical', 'narcissist', 'natural', 'naughty', 'new', 'nimble', 'now', 'objective', 'old', 'optimistic', 'organic', 'paradise', 'party', 'passion', 'passive', 'peace', 'peaceful', 'personal', 'playful', 'pleasing', 'possible', 'powerful', 'preceding', 'predatory', 'prime', 'probable', 'productive', 'professional', 'profit', 'progress', 'public', 'pure', 'radical', 'railway', 'rain', 'real', 'rebellious', 'recession', 'reconciliation', 'recovery', 'relaxed', 'reliability', 'retro', 'rich', 'risk', 'rococo', 'romantic', 'royal', 'rustic', 'sad', 'sadness', 'safe', 'sarcasm', 'secure', 'sensible', 'sensual', 'sex', 'shabby', 'silly', 'simple', 'slow', 'smart', 'smooth', 'snorkel', 'soft', 'solar', 'sold', 'solid', 'somber', 'spiffy', 'sport', 'spring', 'stability', 'star', 'strong', 'studio', 'style', 'stylish', 'submit', 'suburban', 'success', 'summer', 'sun', 'sunny', 'surprise', 'sweet', 'symbol', 'tasty', 'therapeutic', 'threat', 'time', 'tomorrow', 'treason', 'trust', 'trustworthy', 'uncertain', 'uniform', 'unlikely', 'unsafe', 'urban', 'value', 'vanity', 'victorian', 'vitamin', 'vulnerability', 'vulnerable', 'war', 'warm', 'winter', 'wise', 'wish', 'work', 'worm', 'young'];
    var shorthand = {"adjective": "[adj]", "noun": "[n]", "adverb": "[adv]"}

    // add palette swatch colors
    $("div.result-color").each(function (i, e) {
        color = $(e).children("p").html();
        $(e).children(".swatch").css("background-color", color);
    })

    // energy slider active
    colorLabel();
    $("#energyInput").change(function () {
        $(".e-label").removeClass("active");
        colorLabel();
    })

    // scroll to results
    if ($("#results").length > 0) {
        $('html, body').animate({
            scrollTop: $("#results").offset().top
        });
    }
    if ($(".noresult").length > 0) {
        console.log("hi?");
        $('html, body').animate({
            scrollTop: $(".noresult").offset().top
        });
    }

    // gradient
    $(".gradient").each(function (i, e) {
        id = $(e).data("id");
        gradient = "linear-gradient(to right";
        $(".palette[data-id="+id+"]").children(".color").each( function (i, c) {

            color = $(c).children("p").html();
            gradient += ", " + color;
        });
        gradient += ")";
        $(e).css("background-image", gradient)
    });

    // breakdown bars
    $(".bar").each(function (i, e) {
        percent = $(e).data("percent");
        $(e).css("width", percent*210);
    })

    // show breakdown
    $(".more-info").hover(function(i) {
       $(".breakdown[data-value=" + $(i.target).data("value") + "]").removeClass("hidden");
    }, function(i) {
        if($(i.target).attr("data-event") != "click") {
            $(".breakdown[data-value=" + $(i.target).data("value") + "]").addClass("hidden");
        }
    });
    $("html").click(function(e) {
        // hide breakdown when clicking outside of it
        if (e.target.class != "breakdown" && $("[data-event='click']").length > 0) {
            let val = $("[data-event='click']").data("value");
            $(".breakdown[data-value=" + val + "]").addClass("hidden");
            $("[data-event='click']").removeAttr("data-event");
        }
        // show breakdown when more info button is clicked
        if (e.target.classList.contains("more-info")) {
            $(".breakdown[data-value=" + $(e.target).data("value") + "]").removeClass("hidden");
            $(e.target).attr("data-event", "click");
        }
    });

    // tooltip (bootstrap)
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl)
    })

    // add color
    $("#addColorBtn").on("click", function(e) {
        $(".picker").addClass("hidden");
        // show color 2
        if ($(".color-input.hidden").length == 1) {
            let c = $(".color-input.hidden").attr("id") == "color1" ? 1 : 2;
            $("#swatch" + c).css("padding-top", "40px");
            $("#swatch" + c).html("Click here to activate color picker.");
            $("#swatch" + c).css("background-color", "white");
            $("#color" + c + "Input").val("");
            $("#color" + c).removeClass("hidden");
            $("#addColorBtn").addClass("hidden");
        }

        // show color 1
        if ($(".color-input.hidden").length == 2) {
            $("#swatch1").css("background-color", "white");
            $("#swatch1").css("padding-top", "40px");
            $("#swatch1").html("Click here to activate color picker.");
            $("#color1").removeClass("hidden");
            $("#color1Input").val("");
        }
    })


    // color picker 1
    var colorPicker1 = new iro.ColorPicker('#picker1', {
        width: 200,
        wheelLightness: false
    });
    colorPickerFunc(1, colorPicker1);

    // color picker 2
    var colorPicker2 = new iro.ColorPicker('#picker2', {
        width: 200,
        wheelLightness: false,
    });
    colorPickerFunc(2, colorPicker2);

    // sticky color
    $(".color-input:not(.hidden)").each( function (i, e) {
        let swatch = $(e).find(".necessary-swatch");
        let hex = $(e).find("input").val();
        console.log(swatch);
        let colorpicker = $(e).find("button").attr("value") == 1 ? colorPicker1 : colorPicker2;
        if (hex.match(hexRegex)) {
            $(swatch).css("background-color", "#"+hex);
            colorpicker.setColors([hex]);
        } else {
            $(swatch).html("?");
            $(swatch).css("background-color", "white");
            $(swatch).css("padding-top", "60px");
            colorpicker.setColors(["#ffffff"]);
        }
    });

    // add color btn
    if ($(".color-input:not(.hidden)").length == 2) {
        $("#addColorBtn").addClass("hidden");
    }
    if ($(".color-input:not(.hidden)").length == 1) {
        $("#addColorBtn").removeClass("hidden");
    }

    // deleting color
    $(".close-swatch").on("click", function(e) {
        console.log("closing swatch");
        color = $(e.target).attr("value");
        let colorpicker = color == 1 ? colorPicker1 : colorPicker2;
        removeColor(color, colorpicker);
    })
    
    // BUG: when pressing enter in hex input it submits?
    $("form input").keydown(function (e) {
        console.log("keydown");
        if (e.keyCode == 13) {
            e.preventDefault();
            return false;
        }
    });     

    $("#clearBtn").on("click", function () {
        $("#tagsInput").attr("value","");
        $("#tagsInput").val("");
        $select[0].selectize.clear();
        $select[0].selectize.clearOptions();

        $("#invalidWords").val("");
        $("#multiDefWords").val("");

        // energy
        $(".e-label").removeClass("active");
        $("#energyInput").val(5);
        colorLabel();

        // necessary colors
        removeColor(1, colorPicker1);
        removeColor(2, colorPicker2);
    });

    // keyword selectize input
    var xhr;
    $select = $("#tagsInput").selectize({
        plugins: ["remove_button", "restore_on_backspace"],
        delimiter: ",",
        persist: false,
        render: {
            item: function (item, escape) {
                let word = escape(item.value)
                if (cymbolism.includes(item.value)) {
                    return "<div class='item' id='w-" + word + "'>" +
                        '<span class="name">' + word + '</span>' + "</div>";
                } else {
                    let definitions = [];
                    let valid = false;
                    xhr && xhr.abort();
                    let keywordURL = "https://wordsapiv1.p.rapidapi.com/words/" + word + "/definitions";
                    xhr = $.ajax({
                        url: keywordURL,
                        headers: {
                            'x-rapidapi-host': 'wordsapiv1.p.rapidapi.com',
                            'x-rapidapi-key': 'e18e293e71msh899aee87d7816d4p1d4574jsn248bb0b951da'
                        },
                        async: false,
                        success: function (results) {
                            // filter out verbs
                            definitions = results["definitions"].filter( function(d) {
                                return d["partOfSpeech"] != "verb";
                            });
                            if (definitions.length > 1) {
                                valid = true;
                                $("#options").append("<div id='" + word + "-options' class='mt-3'></div>");
                                $("#" + word + "-options").append("<label class='options-label' for='" + word + "'><strong>" + word + "</strong></label><br>");
                                definitions.forEach(d => {
                                    console.log(d["partOfSpeech"]);
                                    let radio = '<div class="radio-input d-flex flex-row">' +
                                        '<input class="radio-button" type="radio" name=' + word + ' value=' + d["definition"].replaceAll(" ", "%") + '>' + 
                                        '<p class="def-text mb-1 ms-2">' + (d["partOfSpeech"] ? shorthand[d["partOfSpeech"]] : "") + " " + d["definition"] + '</p>' +
                                    '</input></div>';
                                    $("#" + word + "-options").append(radio);
                                })

                                let multiDefs = $("#multiDefWords").val();
                                console.log("generating multi defs");
                                console.log(multiDefs);
                                if (multiDefs == "") {
                                    $("#multiDefWords").val(word);
                                } else {
                                    $("#multiDefWords").val(multiDefs + "," + word);
                                }
                                console.log($("#multiDefWords").val());
                            } else {
                                $("#multiDefWords").removeAttr('value');
                                if (definitions.length == 1) {
                                    valid = true;
                                }
                            }
                        },
                        error: function () {
                            console.log("invalid word");
                            let curr = $("#invalidWords").val();
                            if (curr == "") {
                                $("#invalidWords").val(word);
                            } else {
                                $("#invalidWords").val(curr + "," + word);
                            }
                            console.log($("#invalidWords").val());
                        }
                    });
                    return "<div class='" + (valid ? "" : "invalid-word") + " item' id='w-" + word + "'>" +
                        '<span class="name">' + word + '</span>' +
                        "</div>";
                }
            }
        },
        create: function(input) {
            return {
                value: input,
                text: input
            }
        }
    });
    
    // remove from multiDefs when removing item
    var selectize = $select[0].selectize;
    selectize.on("item_remove", function (e) {
        console.log("selectize remove");
        console.log(e);
        selectize.removeOption(e);
        let multiDefs = $("#multiDefWords").val()
        if (multiDefs) {
            let multiDefList = multiDefs.split(",");
            let i = multiDefList.indexOf(e);
            let res = "";
            if (i != -1) {
                for (j = 0; j < multiDefList.length; j++) {
                    if (i != j) {
                        if (res != "") { res += "," }
                        res += multiDefList[j];
                    }
                }
                $("#multiDefWords").val(res);
                console.log(res);
            }
        }
    });
})


function colorLabel() {
    let energyId = "#e" + $("#energyInput").val();
    $(energyId).addClass("active");
}

function createWord(arg) {
    let txt = $(arg).html();
    let i = txt.indexOf(" - ");
    let selectize = $keywords[0].selectize;
    selectize.removeItem(txt.substring(0, i));
    selectize.createItem(txt);
}

function closeModal() {
    $("#background").addClass("hidden");
}

function showBreakdown(show, i) {
    if (show) {
        $(".breakdown[data-value" + i + "]").removeClass("hidden");
    } else {
        $(".breakdown[data-value" + i + "]").addClass("hidden");
    }
}

function removeColor(c, colorpicker) {
    $("#picker" + c).addClass("hidden");
    $("#color" + c).addClass("hidden");

    // reset color
    $("#color" + c + "Input").val("");
    $("#swatch" + c).css("background-color","white");
    $("#swatch" + c).css("padding-top", "40px");
    $("#swatch" + c).html("Click here to activate color picker.");
    console.log("resetting color");
    colorpicker.setColors(["#ffffff"]);

    // show add button?
    if ($(".color-input.hidden").length < 2) {
        console.log("removed, showing add button");
        $("#addColorBtn").removeClass("hidden");
    }
}
String.prototype.replaceAt = function(index, replacement) {
    if (index >= this.length) {
        return this.valueOf();
    }
 
    var chars = this.split('');
    chars[index] = replacement;
    return chars.join('');
}
function vote2(voteNum, palette, pID) {

    var currVote = 9
    var currVal = document.getElementById("vote").value

    if (currVal == ""){
      var addVote = palette + " " + voteNum;
      currVal = addVote
      currVote = voteNum
      
    }
    else if (currVal.includes(palette)){
        var paletteSize = 0
        var paletteStr = String(palette)
        var paleteIndex = currVal.indexOf(paletteStr);
        console.log(paleteIndex)
        palette.forEach(pal => {
            paletteSize += pal.length
        });
        paleteIndex += paletteSize + 5;
        if (currVal.charAt(paleteIndex) == '-'){
            if (Number(voteNum) == 1) {
                var addVote = ":" + palette + " -1"
                currVal = currVal.replace(addVote, "");
                addVote = palette +  " -1"
                currVal = currVal.replace(addVote, "");
        
                currVote = voteNum
                currVote = 0 
            }
            else {currVote = -1}
        }
        else if (currVal.charAt(paleteIndex) == '1'){
            if (Number(voteNum) == -1) {
                var addVote = ":" + palette + " 1"
                currVal = currVal.replace(addVote, "");
                addVote =  palette + " 1";
                currVal = currVal.replace(addVote, "");
        
                currVote = voteNum
                currVote = 0 
            }
            else {currVote = 1}
        }
    }
    else{
      var addVote = ":" + palette + " " + voteNum;
      currVal += addVote;
      currVote = voteNum
      // up change class to update

      //change down 
    }
    document.getElementById("vote").value = currVal;

    pID = "p" + pID
    console.log(pID)
    document.getElementById(pID).textContent = currVote;

}
function openPicker(open, p, type, paletteId) {
    if (open) {
        $(".picker").addClass("hidden");
        $("#picker" + p).removeClass("hidden");
    } else {
        $("#picker" + p).addClass("hidden");
    }
}

function colorPickerFunc(c, colorpicker) {
    let input = "#color" + c + "Input";
    let swatch = "#swatch" + c;
    // change swatch when picker changes
    colorpicker.on('color:change', function(color) {
        $(swatch).html("");
        color = color.hexString;
        $(input).val(color.substring(1,color.length));
        $(input).attr("value", color.substring(1,color.length));
        $(swatch).css("background-color", color);
    });

    // set color when user inputs hex code into picker 
    $(input).on("change", function(e) {
        let hex = $(input).val();
        $(input).attr("value", hex);
        if (hex.match(hexRegex)) {
            $(swatch).html("");
            console.log(["#" + hex]);
            let lst = ["#" + hex];
            colorpicker.setColors(lst);
            $(swatch).css("background-color", "#" + hex);
        } else {
            console.log("invalid color");
            $(swatch).html("?");
            $(swatch).css("padding-top", "60px");
            $(swatch).css("background-color", "white");
            colorpicker.setColors(["#ffffff"]);
        }
    })
}

function copyToClipboard(value) {
    var $temp = $("<input>");
    $("body").append($temp);

    $temp.val(value).select();
    document.execCommand('copy');
    $temp.remove();
}

function copyHex(hex, copy) {
    copyToClipboard(hex);

    // tooltip
    let tooltip = $(copy);
    setTimeout(function(){ tooltip.tooltip("hide"); }, 500);
}

function copyPalette(palette, tip) {
    copyToClipboard(palette.map(i => '#' + i).join(", "));
    console.log(palette.map(i => '#' + i).join(", "));

    let tooltip = $(tip);
    setTimeout(function(){ tooltip.tooltip("hide"); }, 500);
}

function loading() {
    console.log("loading");
    $("#loader").css("display", "flex");
    $("#content").css("display", "none");
}
