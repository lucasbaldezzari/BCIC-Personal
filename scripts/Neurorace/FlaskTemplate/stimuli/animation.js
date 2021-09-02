(function() {
  var demo, run;

  demo = $("#whole-thing");

  run = function() {
    // Imagine a box

    // The box has border-top
    setTimeout(function() {
      return demo.addClass("step-1");
    }, 2500);
    
    // It also has the other borders
    setTimeout(function() {
      return demo.addClass("step-2");
    }, 5000);
    setTimeout(function() {
      return demo.addClass("step-3");
    }, 5500);
    setTimeout(function() {
      return demo.addClass("step-4");
    }, 6000);
    
    // Notice how the borders meet each other at angles.
    setTimeout(function() {
      return demo.addClass("step-5");
    }, 7500);
    
    // The background of the box is transparent.
    setTimeout(function() {
      return demo.addClass("step-6");
    }, 10000);
    
    // The box is actually zero width and zero height.
    setTimeout(function() {
      return demo.addClass("step-7");
    }, 12000);
    
    // Three of the borders are actually transparent in color.
    setTimeout(function() {
      return demo.addClass("step-8");
    }, 14000);
    setTimeout(function() {
      return demo.addClass("step-9");
    }, 14500);
    setTimeout(function() {
      return demo.addClass("step-10");
    }, 15000);
    
    //# Done
    return setTimeout(function() {
      return demo.addClass("step-11");
    }, 18000);
  };

  run();

  $("#re-run").on('click', function() {
    $("#whole-thing").removeClass();
    return run();
  });

  window.__run = run;

}).call(this);