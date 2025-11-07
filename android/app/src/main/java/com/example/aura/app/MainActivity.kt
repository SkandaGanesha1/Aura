package com.example.aura.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.viewModels
import androidx.lifecycle.lifecycleScope
import com.example.aura.app.databinding.ActivityMainBinding
import com.example.aura.aura_core.AuraViewModel
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {

    private lateinit var binding: ActivityMainBinding
    private val viewModel: AuraViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.executeButton.setOnClickListener {
            val intentText = binding.intentInput.text?.toString().orEmpty()
            if (intentText.isBlank()) {
                binding.statusText.text = getString(R.string.intent_missing)
                return@setOnClickListener
            }
            lifecycleScope.launch {
                val result = viewModel.handleIntent(intentText)
                binding.statusText.text = result
            }
        }

        viewModel.state.observe(this) { state ->
            binding.statusText.text = state.statusMessage
        }
    }
}
