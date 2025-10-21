class Chessboard {
    constructor(node) {
        /** @type {HTMLTableElement} */
        this.node = node
        this.init()
    }

    init() {
        this.node.innerHTML = ""
        for (let y=0; y<8; y++) {
            const row = this.node.insertRow()
            for (let x=0; x<8; x++) {
                const cell = row.insertCell()
                cell.dataset.x == x
                cell.dataset.y == y
            }
        }
    }
}


window.addEventListener("load", () => {
    const board = new Chessboard(document.getElementById("chessboard"))
})