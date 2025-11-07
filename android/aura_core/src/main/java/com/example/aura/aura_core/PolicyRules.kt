package com.example.aura.aura_core

object PolicyRules {
    private val sensitiveKeywords = setOf("transfer", "payment", "delete", "purchase")

    fun requiresUserConfirmation(intent: String): Boolean {
        return sensitiveKeywords.any { keyword ->
            intent.contains(keyword, ignoreCase = true)
        }
    }
}
