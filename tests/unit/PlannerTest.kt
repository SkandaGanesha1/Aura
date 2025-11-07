package com.example.aura.aura_core

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class PolicyRulesTest {

    @Test
    fun sensitiveKeywordsRequireConfirmation() {
        assertTrue(PolicyRules.requiresUserConfirmation("Please transfer $20"))
        assertFalse(PolicyRules.requiresUserConfirmation("Send a friendly reminder"))
    }
}
