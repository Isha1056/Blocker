// @author Rob W <http://stackoverflow.com/users/938089/rob-w>
// Demo: var serialized_html = DOMtoString(document);

function DOMtoString(document_root) {
    const xhr = new XMLHttpRequest();
    var html = '',
        node = document_root.firstChild;
    while (node) {
        switch (node.nodeType) {
        case Node.ELEMENT_NODE:
            html += node.outerHTML;
            break;
        case Node.TEXT_NODE:
            html += node.nodeValue;
            break;
        case Node.CDATA_SECTION_NODE:
            html += '<![CDATA[' + node.nodeValue + ']]>';
            break;
        case Node.COMMENT_NODE:
            html += '<!--' + node.nodeValue + '-->';
            break;
        case Node.DOCUMENT_TYPE_NODE:
            // (X)HTML documents are identified by public identifiers
            html += "<!DOCTYPE " + node.name + (node.publicId ? ' PUBLIC "' + node.publicId + '"' : '') + (!node.publicId && node.systemId ? ' SYSTEM' : '') + (node.systemId ? ' "' + node.systemId + '"' : '') + '>\n';
            break;
        }
        node = node.nextSibling;
    }
    const json = {
        "title": html,
        "status": "recieved"
    };
    xhr.open('POST', 'http://localhost:5000/');
    xhr.setRequestHeader('Content-Type', 'application/json');
    xhr.send(JSON.stringify(json));
    xhr.addEventListener('load', reqListener);
    return html;
}

function getNodesThatContain(text) {
    var textNodes = $(document).find(":not(iframe, script, style, noscript, title, object)")
      .contents().filter( 
          function() {
           return this.nodeType == 3 
             && this.textContent.indexOf(text) > -1;
    });
    
    return textNodes.parent();
};

function reqListener() {
    //console.log('this.responseText:', typeof(this.responseText));
    //console.log('this.status:', this.status);
    var r = JSON.parse(this.responseText)['response'];
    //console.log(r['response']);
    for( let i=0; i<r.length; i++) {
        var x = getNodesThatContain(r[i]);
        //console.log(x);
        for( let j=0; j<r.length; j++) {
            if (typeof x[j] !== 'undefined'){
                console.log(x[j]);
                $(x[j]).addClass("blurmytext");
            }
        }
    }
}

chrome.runtime.sendMessage({
    action: "getSource",
    source: DOMtoString(document)
});

