$(document).ready(function () {
    var cymbolism = ['abuse', 'accessible', 'addiction', 'agile', 'amusing', 'anger', 'anticipation', 'art deco', 'authentic', 'authority', 'average', 'baby', 'beach', 'beauty', 'beer', 'benign', 'bitter', 'blend', 'blissful', 'bold', 'book', 'boss', 'brooklyn', 'busy', 'calming', 'capable', 'car', 'cat', 'certain', 'charity', 'cheerful', 'chicago', 'classic', 'classy', 'clean', 'cold', 'colonial', 'comfort', 'commerce', 'compelling', 'competent', 'confident', 'consequence', 'conservative', 'contemporary', 'cookie', 'corporate', 'cottage', 'crass', 'creative', 'cute', 'dance', 'dangerous', 'decadent', 'decisive', 'deep', 'devil', 'discount', 'disgust', 'dismal', 'dog', 'drunk', 'dublin', 'duty', 'dynamic', 'earthy', 'easy', 'eclectic', 'efficient', 'elegant', 'elite', 'enduring', 'energetic', 'entrepreneur', 'environmental', 'erotic', 'excited', 'expensive', 'experience', 'fall', 'familiar', 'fast', 'fear', 'female', 'football', 'freedom', 'fresh', 'friendly', 'fun', 'furniture', 'future', 'gay', 'generic', 'georgian', 'gloomy', 'god', 'good', 'goth', 'government', 'grace', 'great', 'grow', 'happy', 'hard', 'hate', 'hazardous', 'hippie', 'hockey', 'honor', 'hope', 'hot', 'hunting', 'hurt', 'hygienic', 'ignorant', 'imagination', 'impossible', 'improbable', 'influence', 'influential', 'insecure', 'inviting', 'invulnerable', 'jacobean', 'jealous', 'joy', 'jubilant', 'junkie', 'knowledge', 'kudos', 'launch', 'lazy', 'leader', 'liberal', 'library', 'light', 'likely', 'lonely', 'love', 'magic', 'marriage', 'maximal', 'mean', 'medicine', 'melancholy', 'mellow', 'minimal', 'mission', 'modern', 'moment', 'money', 'music', 'mystical', 'narcissist', 'natural', 'naughty', 'new', 'nimble', 'now', 'objective', 'old', 'optimistic', 'organic', 'paradise', 'party', 'passion', 'passive', 'peace', 'peaceful', 'personal', 'playful', 'pleasing', 'possible', 'powerful', 'preceding', 'predatory', 'prime', 'probable', 'productive', 'professional', 'profit', 'progress', 'public', 'pure', 'radical', 'railway', 'rain', 'real', 'rebellious', 'recession', 'reconciliation', 'recovery', 'relaxed', 'reliability', 'retro', 'rich', 'risk', 'rococo', 'romantic', 'royal', 'rustic', 'sad', 'sadness', 'safe', 'sarcasm', 'secure', 'sensible', 'sensual', 'sex', 'shabby', 'silly', 'simple', 'slow', 'smart', 'smooth', 'snorkel', 'soft', 'solar', 'sold', 'solid', 'somber', 'spiffy', 'sport', 'spring', 'stability', 'star', 'strong', 'studio', 'style', 'stylish', 'submit', 'suburban', 'success', 'summer', 'sun', 'sunny', 'surprise', 'sweet', 'symbol', 'tasty', 'therapeutic', 'threat', 'time', 'tomorrow', 'treason', 'trust', 'trustworthy', 'uncertain', 'uniform', 'unlikely', 'unsafe', 'urban', 'value', 'vanity', 'victorian', 'vitamin', 'vulnerability', 'vulnerable', 'war', 'warm', 'winter', 'wise', 'wish', 'work', 'worm', 'young'];
    var shorthand = {"adjective": "[adj]", "noun": "[n]", "adverb": "[adv]"}

    // add palette swatch colors
    $("div.color").each(function (i, e) {
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
    // $("[data-event='click']").removeAttr("data-event");
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
                            console.log("hello???");
                            console.log(results);
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