package com.example.bert

import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.POST

interface SimpleApi{
	@POST("predict")
	suspend fun  pushMail(
		@Body mail :Mail
	): Response<MailResponse>
}