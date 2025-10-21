const Piece = {
    King: "k",
    Queen: "q",
    Bishop: "b",
    Knight: "n",
    Rook: "r",
    Pawn: "p",

    isWhite: p => p.toUpperCase() === p,
    isBlack: p => p.toLowerCase() === p
}

function make2DArray(height, width, value=0) {
    const arr = []
    for (let y=0; y<height; y++) {
        const row = []
        for (let x=0; x<width; x++) {
            row.push(value)
        }
        arr.push(row)
    }
    return arr
}

class Chessboard {
    static DEFAULT_STATE = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    constructor(node) {
        /** @type {HTMLDivElement} */
        this.node = node
        this.state = make2DArray(8, 8, "")
        this.init()
        this.renderPieces()
    }

    init() {
        this.loadFEN(Chessboard.DEFAULT_STATE)
        this.node.innerHTML = ""
        for (let y=0; y<8; y++) {
            for (let x=0; x<8; x++) {
                const cell = document.createElement("div")
                cell.classList.add("tile")
                cell.dataset.x = x
                cell.dataset.y = y

                if ((x + y) % 2 == 0) cell.classList.add("light")
                this.node.appendChild(cell)
            }
        }
    }

    /**
     * 
     * @param {string} fen 
     */
    loadFEN(fen) {
        let parts = fen.split(" ")
        const state = make2DArray(8, 8, "")
        let x = 0
        let y = 0
        for (let i=0; i<parts[0].length; i++) {
            const char = parts[0][i]
            if (char === "/") {
                x = 0
                y += 1
            } else if (!isNaN(+char)) {
                x += +char
            } else {
                state[y][x] = char
                x += 1
            }
        }
        this.state = state
    }

    renderPieces() {
        for (let y=0; y<8; y++) {
            for (let x=0; x<8; x++) {
                const piece = this.state[y][x]
                if (!piece) continue
                const kind = piece.toLowerCase()
                const isBlack = Piece.isBlack(piece)
                const elem = document.createElement("img")
                elem.src = `/images/${kind}.png`
                elem.classList.add("piece")
                if (isBlack) elem.classList.add("black")
                const tile = this.node.childNodes[y * 8 + x]
                tile.appendChild(elem)
            }
        }
    }
}


window.addEventListener("load", () => {
    window.board = new Chessboard(document.getElementById("chessboard"))
})