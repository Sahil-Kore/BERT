package com.example.bert

import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import com.example.bert.ui.theme.BERTTheme
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
	override fun onCreate(savedInstanceState: Bundle?) {
		super.onCreate(savedInstanceState)
		setContent {
			BERTTheme {
				var inputString by remember{ mutableStateOf("") }
				var triggerButton by remember { mutableStateOf(false) }
				var resultString by remember { mutableStateOf("") }
				Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
					Column (modifier = Modifier.padding(innerPadding)){
							TextField(value=inputString, onValueChange = {
								inputString =it
							})
							Button(onClick ={triggerButton=!triggerButton} ){
								Text("Predict")
							}
						if(resultString!="") {
							Text("Result is $resultString")
						}
					}
					LaunchedEffect(triggerButton) {
						CoroutineScope(Dispatchers.IO).launch {
							if(inputString!="") {
								try {
									val request = Mail(inputString)
									val response = RetrofitInstance.api.pushMail(request)
									if (response.isSuccessful && response.body() != null) {
										response.body()?.let {
											resultString = it.Prediction_class
										}
									} else {
										Log.d("Retrofit", "Response is empty")

									}
								} catch (e: Exception) {
									Log.d("Retrofit", e.toString())
								}
							}
						}
					}
				}
			}
		}
	}
}

@Composable
fun Greeting(name: String, modifier: Modifier = Modifier) {
	Text(
		text = "Hello $name!",
		modifier = modifier
	)
}

@Preview(showBackground = true)
@Composable
fun GreetingPreview() {
	BERTTheme {
		Greeting("Android")
	}
}