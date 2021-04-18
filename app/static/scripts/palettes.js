$(document).ready(function() {
    $("div.color").each( function(i, e) {
        color = $(e).children("p").html();
        $(e).children(".swatch").css("background-color", color);
    })
})