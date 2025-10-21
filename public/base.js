const API_URL = "/api"

async function apiPost(path, body) {
    const url = API_URL + (path.startsWith("/") ? "" : "/") + path
    const res = await fetch(url, {
        method: "POST",
        body: JSON.stringify(body),
        headers: {
            "Content-Type": "application/json"
        }
    })

    if (!res) return null

    return await res.json()
}

class Popup {
    constructor(node) {
        this.node = node
        this.visible = false
    }

    show() {
        this.visible = true
        this.node.classList.add("show")
    }

    hide() {
        this.visible = false
        this.node.classList.remove("show")
    }
}