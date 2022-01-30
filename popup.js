/// turn on to see the html in pop up

// chrome.runtime.onMessage.addListener(function(request, sender) {
//     if (request.action == "getSource") {
//       message.innerText = request.source;
//     }
//   });
  var ch=0;
  
  function onWindowLoad() {
  
    var message = document.querySelector('#message');
  
    chrome.tabs.executeScript(null, {
      file: "getPagesSource.js"
    }, function() {
      // If you try and inject into an extensions page or the webstore/NTP you'll get an error
      if (chrome.runtime.lastError) {
        message.innerText = 'There was an error injecting script : \n' + chrome.runtime.lastError.message;
      }
    });
  
  }

  function blurRemoval() {
  
    var message = document.querySelector('#message');
  
    chrome.tabs.executeScript(null, {
      file: "NoBlur.js"
    }, function() {
      // If you try and inject into an extensions page or the webstore/NTP you'll get an error
      if (chrome.runtime.lastError) {
        message.innerText = 'There was an error injecting script : \n' + chrome.runtime.lastError.message;
      }
    });
  
  }

  
  //window.onload = onWindowLoad;
  document.getElementById("#mainslider").addEventListener("click", function() {
    if(ch==0) {
      console.log("XXXXXXXXXXXXXXXXXXXXXXXXXX")
      onWindowLoad();
      ch=1;
    }
    else {
      blurRemoval();
      ch=0;
    }
    
}, false);
