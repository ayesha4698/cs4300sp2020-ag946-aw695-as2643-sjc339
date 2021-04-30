// var keywords;

$(document).ready(function() {
    var cymbolism = ['abuse', 'accessible', 'addiction', 'agile', 'amusing', 'anger', 'anticipation', 'art deco', 'authentic', 'authority', 'average', 'baby', 'beach', 'beauty', 'beer', 'benign', 'bitter', 'blend', 'blissful', 'bold', 'book', 'boss', 'brooklyn', 'busy', 'calming', 'capable', 'car', 'cat', 'certain', 'charity', 'cheerful', 'chicago', 'classic', 'classy', 'clean', 'cold', 'colonial', 'comfort', 'commerce', 'compelling', 'competent', 'confident', 'consequence', 'conservative', 'contemporary', 'cookie', 'corporate', 'cottage', 'crass', 'creative', 'cute', 'dance', 'dangerous', 'decadent', 'decisive', 'deep', 'devil', 'discount', 'disgust', 'dismal', 'dog', 'drunk', 'dublin', 'duty', 'dynamic', 'earthy', 'easy', 'eclectic', 'efficient', 'elegant', 'elite', 'enduring', 'energetic', 'entrepreneur', 'environmental', 'erotic', 'excited', 'expensive', 'experience', 'fall', 'familiar', 'fast', 'fear', 'female', 'football', 'freedom', 'fresh', 'friendly', 'fun', 'furniture', 'future', 'gay', 'generic', 'georgian', 'gloomy', 'god', 'good', 'goth', 'government', 'grace', 'great', 'grow', 'happy', 'hard', 'hate', 'hazardous', 'hippie', 'hockey', 'honor', 'hope', 'hot', 'hunting', 'hurt', 'hygienic', 'ignorant', 'imagination', 'impossible', 'improbable', 'influence', 'influential', 'insecure', 'inviting', 'invulnerable', 'jacobean', 'jealous', 'joy', 'jubilant', 'junkie', 'knowledge', 'kudos', 'launch', 'lazy', 'leader', 'liberal', 'library', 'light', 'likely', 'lonely', 'love', 'magic', 'marriage', 'maximal', 'mean', 'medicine', 'melancholy', 'mellow', 'minimal', 'mission', 'modern', 'moment', 'money', 'music', 'mystical', 'narcissist', 'natural', 'naughty', 'new', 'nimble', 'now', 'objective', 'old', 'optimistic', 'organic', 'paradise', 'party', 'passion', 'passive', 'peace', 'peaceful', 'personal', 'playful', 'pleasing', 'possible', 'powerful', 'preceding', 'predatory', 'prime', 'probable', 'productive', 'professional', 'profit', 'progress', 'public', 'pure', 'radical', 'railway', 'rain', 'real', 'rebellious', 'recession', 'reconciliation', 'recovery', 'relaxed', 'reliability', 'retro', 'rich', 'risk', 'rococo', 'romantic', 'royal', 'rustic', 'sad', 'sadness', 'safe', 'sarcasm', 'secure', 'sensible', 'sensual', 'sex', 'shabby', 'silly', 'simple', 'slow', 'smart', 'smooth', 'snorkel', 'soft', 'solar', 'sold', 'solid', 'somber', 'spiffy', 'sport', 'spring', 'stability', 'star', 'strong', 'studio', 'style', 'stylish', 'submit', 'suburban', 'success', 'summer', 'sun', 'sunny', 'surprise', 'sweet', 'symbol', 'tasty', 'therapeutic', 'threat', 'time', 'tomorrow', 'treason', 'trust', 'trustworthy', 'uncertain', 'uniform', 'unlikely', 'unsafe', 'urban', 'value', 'vanity', 'victorian', 'vitamin', 'vulnerability', 'vulnerable', 'war', 'warm', 'winter', 'wise', 'wish', 'work', 'worm', 'young'];

    // add palette swatch colors
    $("div.color").each( function(i, e) {
        color = $(e).children("p").html();
        $(e).children(".swatch").css("background-color", color);
    })

    // energy slider active
    colorLabel();
    $("#energyInput").change( function() {
        $(".e-label").removeClass("active");
        colorLabel();
    })

    // console.log("invalid");
    // console.log($(".invalid-word"));
    // if ($(".invalid-word").length == 0) {
    //     $("#invalidWords").removeAttr('value');
    // }

    var xhr;
    $("#tagsInput").selectize({
        plugins: ["remove_button", "restore_on_backspace"],
        delimiter: ",",
        persist: false,
        render: {
            item: function(item, escape) {
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
                        success: function(results) {
                            // filter on adjectives + nouns
                            definitions = results["definitions"];
                            if (definitions.length > 1) {
                                valid = true;
                                let divHTML = $("#options").append("<div id='" + word + "-options'></div>")
                                $(divHTML).append("<label for=''>" + word + "</label><br>");
                                definitions.forEach(d => {
                                    let radio = '<input type="radio" name=' + word + ' value=' + d["definition"].replaceAll(" ", "%") + '>  ' + d["definition"] + '</input><br>';
                                    $("#options").append(radio);
                                })

                                let multiDefs = $("#multiDefWords").val();
                                if (multiDefs == "") {
                                    $("#multiDefWords").val(word);
                                    console.log("here");
                                    console.log($("#multiDefWords").val());
                                } else {
                                    $("#multiDefWords").val(multiDefs + "," + word);
                                }
                            } else {
                                $("#multiDefWords").removeAttr('value');
                                if (definitions.length == 1) {
                                    valid = true;
                                }
                            }

                            // if (!valid) {
                            //     $("#invalidWords").val(0);
                            //     console.log("invalidd words exist");
                            // } else {
                            //     $("#invalidWords").val(1);
                            //     console.log("no invalid words");
                            // }
                        },
                        error: function() {
                            console.log("erroring");
                            let curr = $("#invalidWords").val();
                            if (curr == "") {
                                $("#invalidWords").val(word);
                            } else {
                                $("#invalidWords").val(curr + "," + word);
                            }
                        }
                    });
                    return "<div class='" + (valid ? "" : "invalid-word") + " item' id='w-" + word + "'>" + 
                        // (definitions.length != 0 ? "<div class='hidden' name='" + "'>" + definitions[0]["definition"] + "</div>": "") +
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

    // var xhr;
    // $keywords = $("#tagsInput").selectize({
    //     plugins: ["remove_button", "restore_on_backspace"],
    //     delimiter: ",",
    //     persist: false,
    //     maxItems: 5,
    //     render: {
    //         item: function(item, escape) {
    //             console.log("rendering");
    //             console.log(item);
    //             $("#suggestions").html("");
    //             if (cymbolism.includes(item.value)) {
    //                 return "<div class='item' id='w-" + escape(item.value) + "'>" + 
    //                     '<span class="name">' + escape(item.value) + '</span>' + "</div>";
    //             // } else if (item.value.includes(" - ")) {
    //             //     let i = item.value.indexOf(" - ");
    //             //     let word = item.value.substring(0, i);
    //             //     let def = item.value.substring(i+2, item.value.length);
    //             //     return "<div class='item' id='w-" + escape(word) + "'>" + 
    //             //         "<div class='hidden'>" + escape(def) + "</div>" + 
    //             //         '<span class="name">' + escape(word) + '</span>' + 
    //             //     "</div>";
    //             } else if (item.value.includes("[")) {
    //                 let i = item.value.indexOf("[");
    //                 let word = item.value.substring(0, i);
    //                 let def = item.value.substring(i+2, item.value.length-1);
    //                 return "<div class='item' id='w-" + escape(word) + "'>" + 
    //                     "<div class='hidden'>" + escape(def) + "</div>" + 
    //                     '<span class="name">' + escape(word) + '</span>' + 
    //                 "</div>";
    //             } else {
    //                 let definitions = [];
    //                 let one_def = false;
    //                 xhr && xhr.abort();
    //                 let keywordURL = "https://wordsapiv1.p.rapidapi.com/words/" + item.value + "/definitions";
    //                 xhr = $.ajax({
    //                     url: keywordURL,
    //                     headers: {
    //                         'x-rapidapi-host': 'wordsapiv1.p.rapidapi.com',
    //                         'x-rapidapi-key': 'e18e293e71msh899aee87d7816d4p1d4574jsn248bb0b951da'
    //                     },
    //                     async: false,
    //                     success: function(results) {
    //                         // filter on adjectives + nouns
    //                         definitions = results["definitions"];
    //                         if (definitions.length > 1) {
    //                             $("#suggestions").append("<p>Did you mean:</p>");
    //                             // definitions.forEach( e => $("#suggestions").append("<p class='text-primary suggest' onclick=''>" + item.value + " - " + e["definition"] + "</p>"))
    //                             // definitions.forEach( e => $("#suggestions").append("<a class='suggest' onclick='createWord(" + escape(item.value) + ", " + e["definition"] + ")'>" + escape(item.value) + " - " + e["definition"] + "</a><br/>"))
    //                             definitions.forEach( e => $("#suggestions").append("<a class='suggest' onclick='createWord(this)'>" + escape(item.value) + " [" + e["definition"] + "]</a><br/>"));
    //                         } else {
    //                             one_def = true;
    //                         }
    //                     },
    //                     // don't know if we want to alert to user or not
    //                     error: function(error) {
    //                         alert(error);
    //                     }
    //                 });

    //                 return "<div class='" + (one_def ? "" : "invalid-word") + " item' id='w-" + escape(item.value) + "'>" + 
    //                     (one_def ? "<div class='hidden'>" + definitions[0]["definition"] + "</div>": "") +
    //                     '<span class="name">' + escape(item.value) + '</span>' + 
    //                 "</div>";
    //             }
    //         }
    //     },
    //     create: function(input) {
    //         console.log("creating");
    //         return {
    //             value: input,
    //             text: input
    //         }
    //     }
    // });

    // console.log("here");
    // console.log($keywords[0].selectize);
    //when closing tag remove did you means!!!!!!!!!!
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