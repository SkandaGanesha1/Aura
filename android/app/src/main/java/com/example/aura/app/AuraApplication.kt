package com.example.aura.app

import android.app.Application
import android.util.Log
import com.example.aura.common.ExecuTorchRuntime

class AuraApplication : Application() {

    override fun onCreate() {
        super.onCreate()
        try {
            ExecuTorchRuntime.initialize(this)
        } catch (t: Throwable) {
            Log.e(TAG, "Failed to initialise ExecuTorch runtime", t)
        }
    }

    companion object {
        private const val TAG = "AuraApplication"
    }
}
