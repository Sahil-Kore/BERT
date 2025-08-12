package com.example.bert

import com.google.gson.annotations.SerializedName

data class Mail(
	var input_str:String
)

data class MailResponse (
	@SerializedName("Prediction_class")
	var Prediction_class:String
)
