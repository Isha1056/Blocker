// @author Rob W <http://stackoverflow.com/users/938089/rob-w>
// Demo: var serialized_html = DOMtoString(document);

function DOMRemove(document_root) {
    $(".blurmytext").removeClass("blurmytext");
}

chrome.runtime.sendMessage({
    action: "getSource",
    source: DOMRemove(document)
});

