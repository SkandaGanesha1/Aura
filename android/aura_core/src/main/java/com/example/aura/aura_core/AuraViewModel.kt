package com.example.aura.aura_core

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.launch

class AuraViewModel(application: Application) : AndroidViewModel(application) {

    private val planner = Planner(application)
    private val _state = MutableLiveData<PlannerResult>()
    val state: LiveData<PlannerResult> = _state

    fun handleIntent(intent: String): String {
        if (intent.isBlank()) {
            return "No intent provided."
        }

        if (PolicyRules.requiresUserConfirmation(intent)) {
            return "Confirmation required for sensitive intent."
        }

        viewModelScope.launch {
            val result = planner.handleIntent(intent)
            _state.postValue(result)
        }

        return "Planning…"
    }
}
