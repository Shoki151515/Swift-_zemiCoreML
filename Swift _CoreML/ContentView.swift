import SwiftUI
// アプリのメインビューを定義
struct ContentView: View {
    var body: some View {
        // CameraViewを全画面表示
        CameraView()
            .edgesIgnoringSafeArea(.all)
    }
}
