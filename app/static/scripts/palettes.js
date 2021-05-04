$(document).ready(function () {
    var cymbolism = ['abuse', 'accessible', 'addiction', 'agile', 'amusing', 'anger', 'anticipation', 'art deco', 'authentic', 'authority', 'average', 'baby', 'beach', 'beauty', 'beer', 'benign', 'bitter', 'blend', 'blissful', 'bold', 'book', 'boss', 'brooklyn', 'busy', 'calming', 'capable', 'car', 'cat', 'certain', 'charity', 'cheerful', 'chicago', 'classic', 'classy', 'clean', 'cold', 'colonial', 'comfort', 'commerce', 'compelling', 'competent', 'confident', 'consequence', 'conservative', 'contemporary', 'cookie', 'corporate', 'cottage', 'crass', 'creative', 'cute', 'dance', 'dangerous', 'decadent', 'decisive', 'deep', 'devil', 'discount', 'disgust', 'dismal', 'dog', 'drunk', 'dublin', 'duty', 'dynamic', 'earthy', 'easy', 'eclectic', 'efficient', 'elegant', 'elite', 'enduring', 'energetic', 'entrepreneur', 'environmental', 'erotic', 'excited', 'expensive', 'experience', 'fall', 'familiar', 'fast', 'fear', 'female', 'football', 'freedom', 'fresh', 'friendly', 'fun', 'furniture', 'future', 'gay', 'generic', 'georgian', 'gloomy', 'god', 'good', 'goth', 'government', 'grace', 'great', 'grow', 'happy', 'hard', 'hate', 'hazardous', 'hippie', 'hockey', 'honor', 'hope', 'hot', 'hunting', 'hurt', 'hygienic', 'ignorant', 'imagination', 'impossible', 'improbable', 'influence', 'influential', 'insecure', 'inviting', 'invulnerable', 'jacobean', 'jealous', 'joy', 'jubilant', 'junkie', 'knowledge', 'kudos', 'launch', 'lazy', 'leader', 'liberal', 'library', 'light', 'likely', 'lonely', 'love', 'magic', 'marriage', 'maximal', 'mean', 'medicine', 'melancholy', 'mellow', 'minimal', 'mission', 'modern', 'moment', 'money', 'music', 'mystical', 'narcissist', 'natural', 'naughty', 'new', 'nimble', 'now', 'objective', 'old', 'optimistic', 'organic', 'paradise', 'party', 'passion', 'passive', 'peace', 'peaceful', 'personal', 'playful', 'pleasing', 'possible', 'powerful', 'preceding', 'predatory', 'prime', 'probable', 'productive', 'professional', 'profit', 'progress', 'public', 'pure', 'radical', 'railway', 'rain', 'real', 'rebellious', 'recession', 'reconciliation', 'recovery', 'relaxed', 'reliability', 'retro', 'rich', 'risk', 'rococo', 'romantic', 'royal', 'rustic', 'sad', 'sadness', 'safe', 'sarcasm', 'secure', 'sensible', 'sensual', 'sex', 'shabby', 'silly', 'simple', 'slow', 'smart', 'smooth', 'snorkel', 'soft', 'solar', 'sold', 'solid', 'somber', 'spiffy', 'sport', 'spring', 'stability', 'star', 'strong', 'studio', 'style', 'stylish', 'submit', 'suburban', 'success', 'summer', 'sun', 'sunny', 'surprise', 'sweet', 'symbol', 'tasty', 'therapeutic', 'threat', 'time', 'tomorrow', 'treason', 'trust', 'trustworthy', 'uncertain', 'uniform', 'unlikely', 'unsafe', 'urban', 'value', 'vanity', 'victorian', 'vitamin', 'vulnerability', 'vulnerable', 'war', 'warm', 'winter', 'wise', 'wish', 'work', 'worm', 'young'];
    var shorthand = {"adjective": "[adj]", "noun": "[n]", "adverb": "[adv]"}
    var hexRegex = "^([A-Fa-f0-9]{6})$";

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
        $(e).css("width", percent);
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
        }
    });

    // tooltip (bootstrap)
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl)
    })

    // sticky color
    $(".color-input:not(.hidden)").each( function (i, e) {
        let swatch = $(e).find(".necessary-swatch");
        let hex = $(e).find("input").val();
        console.log(swatch);
        if (hex.match(hexRegex)) {
            $(swatch).css("background-color", "#"+hex);
        } else {
            $(swatch).html("?");
            $(swatch).css("background-color", "white");
        }
    });

    if ($(".color-input:not(.hidden)").length == 2 || $(".color-input:not(.hidden)").length == 1) {
        $("#addColorBtn").addClass("hidden");
    }

    // add color
    $("#addColorBtn").on("click", function(e) {
        $(".picker").addClass("hidden");
        // show color 2
        if ($(".color-input.hidden").length == 1) {
            console.log("adding color 2");
            console.log($(".color-input.hidden").attr("id"));
            let c = $(".color-input.hidden").attr("id") == "color1" ? 1 : 2;
            console.log(c);
            $("#swatch" + c).css("font-size", "0.8rem");
            $("#swatch" + c).html("Click here to activate color picker.");
            $("#swatch" + c).css("background-color", "white");
            $("#color" + c + "Input").val("");
            $("#color" + c).removeClass("hidden");
            $("#addColorBtn").addClass("hidden");
        }

        // show color 1
        if ($(".color-input.hidden").length == 2) {
            console.log("adding color 1");
            $("#swatch1").css("background-color", "white");
            $("#swatch1").css("font-size", "0.8rem");
            $("#swatch1").html("Click here to activate color picker.");
            $("#color1").removeClass("hidden");
            $("#color1Input").val("");
        }
    })


    // color picker 1
    var colorPicker1 = new iro.ColorPicker('#picker1', {
        width: 200
    });

    // change swatch when picker 1 changes
    colorPicker1.on('color:change', function(color) {
        if (color.index == 0) {
            $("#swatch1").html("");
            color = color.hexString;
            $("#color1Input").val(color.substring(1,color.length));
            $("#color1Input").attr("value", color.substring(1,color.length));
            $("#swatch1").css("background-color", color);
        }
    });

    // set color when user inputs hex code into picker 1
    $("#color1Input").on("change", function(e) {
        console.log("HERE!!!");
        let hex = $("#color1Input").val();
        $("#color1Input").attr("value", hex);
        if (hex.match(hexRegex)) {
            console.log("match regex");
            $("#swatch1").html("");
            colorPicker1.setColors([hex]);
            $("#swatch1").css("background-color", "#" + hex);
        } else {
            console.log("invalid color");
            $("#swatch1").html("?");
            $("#swatch1").css("font-size", "60px");
            $("#swatch1").css("background-color", "white");
        }
    })

    // color picker 2
    var colorPicker2 = new iro.ColorPicker('#picker2', {
        width: 200
    });

    // change swatch when picker 2 changes
    colorPicker2.on('color:change', function(color) {
        if (color.index == 0) {
            $("#swatch2").html("");
            console.log('color 0 changed!');
            console.log(color.index, color.hexString);
            color = color.hexString;
            $("#color2Input").val(color.substring(1,color.length));
            $("#color2Input").attr("value", color.substring(1,color.length));
            $("#swatch2").css("background-color", color);
        }
    });

    // set color when user inputs hex code into picker 2
    $("#color2Input").on("change", function(e) {
        let hex = $("#color2Input").val();
        $("#color2Input").attr("value", hex);
        if (hex.match(hexRegex)) {
            $("#swatch2").html("");
            colorPicker1.setColors([hex]);
            $("#swatch2").css("background-color", "#" + hex);
        } else {
            console.log("invalid color");
            $("#swatch2").html("?");
            $("#swatch2").css("font-size", "60px");
            $("#swatch2").css("background-color", "white");
        }
    })

    // deleting color
    $(".close-swatch").on("click", function(e) {
        console.log("closing swatch");
        color = $(e.target).attr("value");
        let colorpicker = color == 1 ? colorPicker1 : colorPicker2;
        removeColor(color, colorpicker);
    })
    
    // BUG: when pressing enter in hex input it submits?
    $("form input").keydown(function (e) {
        if (e.keyCode == 13) {
            e.preventDefault();
            return false;
        }
    });    

    // keyword selectize input
    var xhr;
    var $select = $("#tagsInput").selectize({
        plugins: ["remove_button", "restore_on_backspace"],
        delimiter: ",",
        persist: false,
        render: {
            item: function (item, escape) {
                let word = escape(item.value)
                $("#suggestions").html("");
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
                                console.log(d);
                                return d["partOfSpeech"] != "verb";
                            });
                            console.log(definitions);
                            if (definitions.length > 1) {
                                valid = true;
                                $("#options").append("<div id='" + word + "-options' class='mt-3'></div>");
                                $("#" + word + "-options").append("<label class='options-label' for='" + word + "'><strong>" + word + "</strong></label><br>");
                                definitions.forEach(d => {
                                    let radio = '<div class="radio-input d-flex flex-row">' +
                                        '<input class="radio-button" type="radio" name=' + word + ' value=' + d["definition"].replaceAll(" ", "%") + '>' + 
                                        '<p class="def-text mb-1 ms-2">' + shorthand[d["partOfSpeech"]] + " " + d["definition"] + '</p>' +
                                    '</input></div>';
                                    $("#" + word + "-options").append(radio);
                                })

                                let multiDefs = $("#multiDefWords").val();
                                if (multiDefs == "") {
                                    $("#multiDefWords").val(word);
                                } else {
                                    $("#multiDefWords").val(multiDefs + "," + word);
                                }
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
            }
        }
    })
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
    console.log("removing color");
    $("#picker" + c).addClass("hidden");
    $("#color" + c).addClass("hidden");

    // reset color
    $("#color" + c + "Input").val("");
    $("#swatch" + c).css("background-color","white");
    $("#swatch" + c).css("font-size", "0.8rem");
    $("#swatch" + c).html("Click here to activate color picker.");
    colorpicker.setColors(["#ffffff"]);

    // show add button?
    console.log("HERE");
    console.log($(".color-input.hidden"));
    if ($(".color-input.hidden").length < 2) {
        $("#addColorBtn").removeClass("hidden");
    }
}

function openPicker(open, p) {
    if (open) {
        $(".picker").addClass("hidden");
        $("#picker" + p).removeClass("hidden");
    } else {
        $("#picker" + p).addClass("hidden");
    }
}