package com.example.aura.desktop

fun main() {
    println("Aura Desktop Companion starting…")
    val server = ContinuityServer()
    server.start()
    println("Waiting for tasks. Press Ctrl+C to exit.")
    server.awaitTermination()
}
