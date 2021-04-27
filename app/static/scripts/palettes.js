$(document).ready(function() {
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
})

function colorLabel() {
    let energyId = "#e" + $("#energyInput").val();
    $(energyId).addClass("active");
}