//
//  ContentView.swift
//  MyCamera
//
//  Created by Swift-Beginners.
//

import SwiftUI

struct ContentView: View {
    // 撮影した写真を保持する状態変数
    @State var captureImage: UIImage? = nil
    // 撮影画面(sheet)の開閉状態を管理
    @State var isShowSheet = false
    
    var body: some View {
        VStack {
            // スペース追加
            Spacer()
            // 撮影した写真があるとき
            if let captureImage {
                // 撮影写真を表示
                Image(uiImage: captureImage)
                // リサイズ
                    .resizable()
                // アスペクト比（縦横比）を維持して画面に収める
                    .scaledToFit()
            }
            
            // スペース追加
            Spacer()
            // 「カメラを起動する」ボタン
            Button {
                // ボタンをタップしたときのアクション
                // カメラが利用可能かチェック
                if UIImagePickerController.isSourceTypeAvailable(.camera) {
                    print("カメラは利用できます")
                    // カメラが使えるなら、isShowSheetをtrue
                    isShowSheet.toggle()
                } else {
                    print("カメラは利用できません")
                }
                
            } label: {
                // テキスト表示
                Text("カメラを起動する")
                // 横幅いっぱい
                    .frame(maxWidth: .infinity)
                // 高さ50ポイントを指定
                    .frame(height: 50)
                // 文字列をセンタリング指定
                    .multilineTextAlignment(.center)
                // 背景を青色に指定
                    .background(Color.blue)
                // 文字色を白色に指定
                    .foregroundStyle(Color.white)
            } // 「カメラを起動する」ボタンここまで
            // 上下左右に余白を追加
            .padding()
            // sheetを表示
            // isPresentedで指定した状態変数がtrueのとき実行
            .sheet(isPresented: $isShowSheet) {
                // UIImagePickerController（写真撮影）を表示
                ImagePickerView(isShowSheet: $isShowSheet, captureImage: $captureImage)
            } // 「カメラを起動する」ボタンのsheetここまで
            
            // captureImageをアンラップする
            if let captureImage {
                // captureImageから共有する画像を生成する
                let shareImage = Image(uiImage: captureImage)
                // 共有シート
                ShareLink(item: shareImage, subject: nil, message: nil,
                          preview: SharePreview("Photo", image: shareImage)) {
                    // テキスト表示
                    Text("SNSに投稿する")
                    // 横幅いっぱい
                        .frame(maxWidth: .infinity)
                    // 高さ50ポイント指定
                        .frame(height: 50)
                    // 背景を青色に指定
                        .background(Color.blue)
                    // 文字色を白色に指定
                        .foregroundStyle(Color.white)
                    // 上下左右に余白を追加
                        .padding()
                } // ShareLinkここまで
            } // アンラップここまで
        } // VStackここまで
    } // bodyここまで
} // ContentViewここまで

#Preview {
    ContentView()
}

