package com.example.aura.actuator

object SafetyChecker {
    private val restrictedPhrases = setOf(
        "delete",
        "transfer",
        "buy",
        "purchase",
        "payment",
        "bank",
        "credential",
        "password",
        "log out",
    )

    fun isSensitive(description: String): Boolean {
        return restrictedPhrases.any { description.contains(it, ignoreCase = true) }
    }
}
