#include <jni.h>
#include <string>
#include "test-bindings/src/securestorage.h"
#define SECURE_STORAGE_CLASS "com/example/game/SecureStorage"
using namespace std;

JNIEnv* g_env=NULL;
namespace org {
namespace blobstore {

void test() {}

int secureStorageSetJavaEnv( JNIEnv* userenv) 
{
    g_env=userenv;
}

int secureStorageWrite( rust::String userkey2, rust::String uservalue2) {

  string userkey = userkey2.c_str();
  string uservalue = uservalue2.c_str();
    JNIEnv* env = g_env;
  
  string secureStorageClass = SECURE_STORAGE_CLASS;
  jclass activityThreadClass = env->FindClass("android/app/ActivityThread");
  jmethodID currentActivityThreadMethod =
      env->GetStaticMethodID(activityThreadClass, "currentActivityThread",
                             "()Landroid/app/ActivityThread;");
  jobject activityThread = env->CallStaticObjectMethod(
      activityThreadClass, currentActivityThreadMethod);
  jmethodID getApplicationMethod = env->GetMethodID(
      activityThreadClass, "getApplication", "()Landroid/app/Application;");
  jobject context = env->CallObjectMethod(activityThread, getApplicationMethod);
  jclass kotlinClass = env->FindClass(secureStorageClass.c_str());
  jmethodID functionMethod = env->GetStaticMethodID(
      kotlinClass, "writeSecureStorage",
      "(Landroid/content/Context;Ljava/lang/String;Ljava/lang/String;)I");
  jstring key = env->NewStringUTF(userkey.c_str());
  jstring value = env->NewStringUTF(uservalue.c_str());
  jint ret = env->CallStaticIntMethod(kotlinClass, functionMethod, context, key,
                                      value);

  return (int)ret;
}

rust::String secureStorageRead(rust::String userkey) {
    JNIEnv* env = g_env;

  string secureStorageClass = SECURE_STORAGE_CLASS;
  jclass activityThreadClass = env->FindClass("android/app/ActivityThread");
  jmethodID currentActivityThreadMethod =
      env->GetStaticMethodID(activityThreadClass, "currentActivityThread",
                             "()Landroid/app/ActivityThread;");
  jobject activityThread = env->CallStaticObjectMethod(
      activityThreadClass, currentActivityThreadMethod);
  jmethodID getApplicationMethod = env->GetMethodID(
      activityThreadClass, "getApplication", "()Landroid/app/Application;");
  jobject context = env->CallObjectMethod(activityThread, getApplicationMethod);
  jclass kotlinClass = env->FindClass(secureStorageClass.c_str());
  jmethodID functionMethod = env->GetStaticMethodID(
      kotlinClass, "readSecureStorage",
      "(Landroid/content/Context;Ljava/lang/String;)Ljava/util/HashMap;");

  jstring x = env->NewStringUTF(userkey.c_str());
  jobject ret =
      env->CallStaticObjectMethod(kotlinClass, functionMethod, context, x);

  jstring resultkey = env->NewStringUTF("result");
  jstring successkey = env->NewStringUTF("success");
  jstring errorkey = env->NewStringUTF("error");
  jclass mapClass = env->FindClass("java/util/HashMap");
  jmethodID getMethod = env->GetMethodID(
      mapClass, "get", "(Ljava/lang/Object;)Ljava/lang/Object;");

  jstring resultvalue =
      (jstring)env->CallObjectMethod(ret, getMethod, resultkey);
  string resultvaluestring = string(env->GetStringUTFChars(resultvalue, 0));

  jstring successvalue =
      (jstring)env->CallObjectMethod(ret, getMethod, successkey);
  string successvaluestring = string(env->GetStringUTFChars(successvalue, 0));

  jstring errorvalue = (jstring)env->CallObjectMethod(ret, getMethod, errorkey);
  string errorvaluestring = string(env->GetStringUTFChars(errorvalue, 0));

  string finalret = resultvaluestring;
  if ("0" == successvaluestring) { // error
    throw errorvaluestring;
  }

  return rust::String(finalret.c_str());
}

} // namespace blobstore
} // namespace org